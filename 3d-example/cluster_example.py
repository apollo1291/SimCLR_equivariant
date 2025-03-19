import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import numpy as np
import os
import sys

pl.seed_everything(0)


def generate_line_data(n_samples=100, logger=None):
    with torch.no_grad():
        n = 5
        stripes = torch.linspace(-1, 1, n)
        ids = stripes[torch.randint(0, len(stripes), (n_samples,))] 
        y_vals = ids #+ torch.randn(n_samples) * 0.05
        
        # Turn ids to classes using the map
        classes = range(n)
        class_map = {stripes[i].item(): classes[i] for i in classes}
        print(class_map)
        ids = torch.tensor([class_map[id.item()] for id in ids])
        

        

        x_vals = torch.rand(n_samples) * 2 - 1
        X = torch.stack((x_vals, y_vals), dim=1)
        T1 = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.float32)
        X3D = X @ T1.T

        if logger:
            fig, ax = plt.subplots()
            ax.scatter(x_vals.numpy(), y_vals.numpy())
            logger.experiment.add_figure("embs/2d", fig)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X3D[:, 0].numpy(), X3D[:, 1].numpy(), X3D[:, 2].numpy())
            logger.experiment.add_figure("embs/3d", fig)

        return ids, X, X3D


class RepeatDataset(Dataset):
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.total_len = len(dataset) * k

    def __len__(self):
        return self.total_len

    def __getitem__(self, index):
        original_index = index % len(self.dataset)
        return self.dataset[original_index]


class ContrastiveDataset(Dataset):
    def __init__(self, ids, X, X3D, transformation, noise_std=0.05):
        self.ids = ids
        self.X = X
        self.X3D = X3D
        self.noise_std = noise_std

        assert transformation in ["shift", "crop"]
        self.transformation = transformation

    def __len__(self):
        return len(self.X)
    
    def get_augmented_point(self, x3d):

        if self.transformation == "shift":
            new_x = torch.rand(()) * 2 - 1
            shift = (x3d[0] - new_x)
            augmented_x3d = x3d.clone()
            augmented_x3d[0] = new_x
            return augmented_x3d, shift

        elif self.transformation == "crop":
            augmented_x3d = x3d.clone()
            crop_param = torch.rand(()) * x3d[0]
            augmented_x3d[0] = crop_param
            return augmented_x3d, crop_param




    def __getitem__(self, idx):
        """
        Return:
            x3d: original 3D point (tensor of shape [3])
            augmented_x3d: the point with x3d[0] replaced by a random new_x
            shift: the difference (x3d[0] - new_x)
            x2d: the original 2D point (just for color-coding / label).
        """
        with torch.no_grad():
            x3d = self.X3D[idx]
            augmented_x3d, transformation_param = self.get_augmented_point(x3d)
            return x3d, augmented_x3d, transformation_param, self.X[idx], self.ids[idx]


class ContrastiveModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr, alpha):
        """
        alpha: The weight for the supervised contrastive part (controls invariance vs. equivariance).
        """
        super().__init__()
        self.lr = lr
        self.output_dim = output_dim
        self.alpha = alpha

        # Simple single-layer MLP for demonstration
        self.proj = nn.Linear(input_dim + 2, output_dim)

        self.val_originals = []
        self.val_projected = []
        self.example_x3d = None

    def forward(self, x, forward_transform, inverse_transform):     
        """
        x shape: (batch_size, 3)
        shift shape: (batch_size,)
        """
        x_prime = torch.cat([x, torch.zeros(x.size(0), 1, device=x.device), torch.zeros(x.size(0), 1, device=x.device)], dim=1)
        
        if inverse_transform != None: 
            x_params = torch.cat([x, inverse_transform.view(-1, 1), torch.zeros(x.size(0), 1, device=x.device)], dim=1)

        else:
            x_params = torch.cat([x, torch.zeros(x.size(0), 1, device=x.device), forward_transform.view(-1, 1)], dim=1)

        return self.proj(x_prime), self.proj(x_params)
    
    def info_nce_loss(self, z1, z2, label_mask):
        # Negative pairwise distance
        sims = - torch.cdist(z1, z2, p=2)
        sims_1 = sims
        sims_2 = sims.permute(1, 0)

        label_mask_1 = label_mask
        label_mask_2 = label_mask_1.permute(1, 0)

        return 0.5 * (
            (-F.log_softmax(sims_1, dim=-1) * label_mask_1).sum(1).mean()
            + (-F.log_softmax(sims_2, dim=-1) * label_mask_2).sum(1).mean()
        )
    def equivariant_loss(self, z1, predicted_z1, z2, predicted_z2, n=None):
        # equivariant is encoded in dimensions 0 ... n if none it is encoded in the entire vector
        if n == None:
            n = z1.size(1)

        equiv_a = self.info_nce_loss(z1[:][:n], predicted_z1[:][n:], torch.eye(z1[:][n:].shape[0], device=z1.device))
        equiv_b = self.info_nce_loss(predicted_z2[:][n:], z2[:][n:], torch.eye(z2[:][n:].shape[0], device=z2.device))

        return equiv_a + equiv_b
    def training_step(self, batch, batch_idx):
        x3d, augmented_x3d, transfomation, x2d, ids = batch

        # 1) Compute forward passes
        z1, predicted_z2 = self.forward(x3d, transfomation, None)
        z2, predicted_z1 = self.forward(augmented_x3d, None, transfomation)

        # 2) Equivariant part (info-nce on z1 vs predicted_z1 and predicted_z2 vs z2)
        equiv_loss = self.equivariant_loss(z1, predicted_z1, z2, predicted_z2, 1)

        # 3) Supervised Contrastive part (invariant along the 'row' dimension)
        # labels = ids
        # same_row = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # supcon_loss = self.info_nce_loss(z1, z1, same_row)
        inv_loss = self.info_nce_loss(z1, z2, torch.eye(z1.shape[0], device=z1.device))


        loss =  (1 - self.alpha) * equiv_loss + self.alpha * inv_loss

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x3d, _, transfomation, x2d, ids = batch
        z, _ = self.forward(x3d, transfomation, None)
        self.val_originals.append(x2d.detach().cpu())
        self.val_projected.append(z.detach().cpu())

        if self.example_x3d is None:
            self.example_x3d = x3d[0].detach().cpu()

    def on_validation_epoch_end(self):
        # Simple 2D scatter plot of the learned embeddings
        original = torch.cat(self.val_originals, dim=0).numpy()
        projected = torch.cat(self.val_projected, dim=0).numpy()
        self.plot_embeddings(original, projected)
        self.val_originals.clear()
        self.val_projected.clear()
        self.plot_shift_variation()

    def plot_embeddings(self, original, projected):
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = original[:, 1]
        ax.scatter(
            projected[:, 0],
            projected[:, 1],
            c=colors,
            label='Projected 2D',
            alpha=0.1
        )
        ax.set_title(f"2D Representation (alpha={self.alpha})")
        self.logger.experiment.add_figure("embs/proj", fig, global_step=self.current_epoch)
    
    def plot_shift_variation(self):
        shift_values = torch.linspace(-1, 1, steps=100)
        example_x3d = self.example_x3d.to(self.device).unsqueeze(0).repeat(100, 1)
        shift_values = shift_values.to(self.device)

        _, z2_predicted = self.forward(example_x3d, shift_values, None)
        z2_predicted = z2_predicted.detach().cpu().numpy()
        
        shift_abs = shift_values.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(z2_predicted[:, 0], z2_predicted[:, 1],
                             c=shift_abs, cmap="viridis")
        ax.set_title(f"Shift Effect on Predicted Representation (z2)\n(alpha={self.alpha})")
        fig.colorbar(scatter, ax=ax, label="Shift Magnitude")
        self.logger.experiment.add_figure("embs/shift_variation",
                                          fig, global_step=self.current_epoch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)


def get_representations(model, data_loader, device="cuda:0"):
    """
    Passes the entire dataset through the model and
    returns:
       - all_original (the 2D points),
       - all_projected (the 2D embeddings from model),
       - all_shifts (the shift value used for each sample).
    """
    model = model.to(device)
    model.eval()

    all_original = []
    all_projected = []
    all_aug_projected = []
    all_shifts = []
    all_ids = []

    with torch.no_grad():
        for x3d, _, shift, x2d, id in data_loader:
            x3d = x3d.to(device)
            shift = shift.to(device)
            z, aug_z = model(x3d, shift, None)

            # Store results on CPU
            all_original.append(x2d)
            all_projected.append(z.cpu())
            all_aug_projected.append(aug_z.cpu())
            all_shifts.append(shift.cpu())
            all_ids.append(id.cpu())

    # Concatenate
    all_original = torch.cat(all_original, dim=0)
    all_projected = torch.cat(all_projected, dim=0)
    all_aug_projected = torch.cat(all_aug_projected, dim=0)
    all_shifts = torch.cat(all_shifts, dim=0)
    all_ids = torch.cat(all_ids, dim=0)

    return all_original, all_projected, all_aug_projected, all_shifts, all_ids


if __name__ == "__main__":
    
    transformation = sys.argv[1]
    # Generate data once
    batch_size = 64
    ids, X, X3D = generate_line_data(batch_size * 10)
    dataset = ContrastiveDataset(ids, X, X3D, transformation=transformation)

    shared_args = dict(
        batch_size=batch_size,
        num_workers=0,  # adjust as needed
        drop_last=True
    )
    train_loader = DataLoader(RepeatDataset(dataset, 10), shuffle=True, **shared_args)
    val_loader = DataLoader(dataset, shuffle=False, **shared_args)

    # Create a directory for saved models and representations
    os.makedirs("saved_runs", exist_ok=True)

    # Loop over alpha values
    alpha_values = np.arange(0, 1, 0.066)  # 0.0, 0.1, ..., 0.7
    for alpha in alpha_values:
        print(f"\n--- Training with alpha={alpha:.2f} ---\n")
        torch.manual_seed(0)
        model = ContrastiveModel(
            input_dim=3,
            output_dim=2,
            lr=2e-4,
            alpha=alpha
        )
        print(list(model.parameters()))

        logger = TensorBoardLogger(
        save_dir="lightning_logs/",
        name="contrastive_model_logs_with_limited_eq",
        version=f"alpha_with_{transformation}_{alpha:.2f}"
    )
        
        trainer = pl.Trainer(
            devices=[0],  # or [1], depending on your GPU setup
            max_epochs=20,  # reduce or increase as desired
            check_val_every_n_epoch=1,
            log_every_n_steps=5,
            logger=logger
        )
        trainer.fit(model, train_loader, val_loader)

        # After training, get the representations
        original_points, projected_points, aug_projected_points,  shift_values, ids = get_representations(
            model, val_loader, device="cuda:0"
        )

        # Save everything needed to disk
        os.makedirs(f"saved_runs/limited_eq/{transformation}", exist_ok=True)
        save_path = f"saved_runs/limited_eq/{transformation}/model_alpha_{alpha:.2f}.pt"
        torch.save({
            "alpha": alpha,
            "model_state_dict": model.state_dict(),
            "original_points": original_points,     # shape [N, 2]
            "augmented_projected_points": aug_projected_points, # shape [N, 2]
            "projected_points": projected_points,   # shape [N, 2]
            "shifts": shift_values,                 # shape [N]
            "ids": ids, # shape [N]
            "X": X,      # entire set of original 2D points
            "X3D": X3D   # entire set of 3D points
        }, save_path)

        print(f"Saved model and representations for alpha={alpha:.2f} to {save_path}")
