import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

pl.seed_everything(0)


def generate_line_data(n_samples=100, logger=None):
    with torch.no_grad():
        n = 5
        stripes = torch.linspace(-1, 1, n)
        ids = stripes[torch.randint(0, len(stripes), (n_samples,))] 
        y_vals = ids + torch.randn(n_samples) * 0.05
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

        return X, X3D


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
    def __init__(self, X, X3D, noise_std=0.05):
        self.X = X
        self.X3D = X3D
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        with torch.no_grad():
            x3d = self.X3D[idx]
            new_x = torch.rand(()) * 2 - 1
            shift = (x3d[0] - new_x)
            augmented_x3d = x3d.clone()
            augmented_x3d[0] = new_x
            return x3d, augmented_x3d, shift, self.X[idx]


class ContrastiveModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, lr, eq):
        super().__init__()
        self.lr = lr
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim + 2, output_dim)

        self.eq = eq
        self.val_originals = []
        self.val_projected = []

        self.example_x3d = None

    def forward(self, x, shift1, shift2):     
        x_prime = torch.cat([x, torch.zeros(x.size(0), 1).to("cuda:1"), torch.zeros(x.size(0), 1).to("cuda:1")], dim=1)
        x_params = torch.cat([x, shift1.view(-1, 1), shift2.view(-1, 1)], dim=1)
        return self.proj(x_prime), self.proj(x_params)

    def info_nce_loss(self, z1, z2, label_mask):
        # sims = torch.einsum("ac,bc->ab", z1, z2)
        sims = - torch.cdist(z1, z2, p=2)

        sims_1 = sims
        sims_2 = sims.permute(1, 0)

        label_mask_1 = label_mask
        label_mask_2 = label_mask_1.permute(1, 0)

        return 1 / 2 * (-F.log_softmax(sims_1, dim=-1) * label_mask_1).sum(1).mean() + \
            1 / 2 * (-F.log_softmax(sims_2, dim=-1) * label_mask_2).sum(1).mean()

    def training_step(self, batch, batch_idx):
        x3d, augmented_x3d, shift, x2d = batch



        # #z1 = self.forward(x3d)

        # # equivariant
        alpha = 1
        z1, predicted_z2 = self.forward(x3d, shift, torch.zeros(shift.shape))
        z2, predicted_z1 = self.forward(augmented_x3d, torch.zeros(shift.shape), shift)
        a = self.info_nce_loss(z1, predicted_z1, torch.eye(z1.shape[0], device=z1.device))
        b = self.info_nce_loss(predicted_z2, z2, torch.eye(z2.shape[0], device=z2.device))
        c = self.info_nce_loss(z1, z2, torch.eye(z1.shape[0], device=z1.device))
        loss = a + b + alpha * c

        # regular Contrastive
        # z2, _ = self.forward(augmented_x3d, -shift)
        # loss = self.info_nce_loss(z1, z2, torch.eye(z1.shape[0], device=z1.device))

        # Mix
        # labels = x2d[:, 1]
        # same_row = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(torch.float32)
        # z2 = self.forward(augmented_x3d)
        # loss = self.info_nce_loss(z1, z2, same_row)

        # SupCon
        # labels = x2d[:, 1]
        # same_row = (labels.unsqueeze(0) == labels.unsqueeze(1)).to(torch.float32)
        # loss = self.info_nce_loss(z1, z1, same_row)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x3d, augmented_x3d, shift, x2d = batch
        if self.example_x3d == None:
            self.example_x3d = x3d[0]
        z, _ = self.forward(x3d, shift, torch.zeros(shift.shape))
        self.val_originals.append(x2d.detach().cpu())
        self.val_projected.append(z.detach().cpu())

    def on_validation_epoch_end(self):
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
            alpha=0.1)
        ax.set_title("2D Representation")
        self.logger.experiment.add_figure("embs/proj", fig, global_step=self.current_epoch)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
    
    def plot_shift_variation(self):
        shift_values = torch.linspace(-1, 1, steps=100)
        example_x3d = self.example_x3d.to(self.device).unsqueeze(0).repeat(100, 1)
        shift_values = shift_values.to(self.device)

        _, z2_predicted = self.forward(example_x3d, shift_values)
        z2_predicted = z2_predicted.detach().cpu().numpy()
        
        shift_abs = shift_values.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(z2_predicted[:, 0], z2_predicted[:, 1], c=shift_abs, cmap="viridis")
        ax.set_title("Shift Effect on Predicted Representation (z2)")
        fig.colorbar(scatter, ax=ax, label="Shift Magnitude")
        self.logger.experiment.add_figure("embs/shift_variation", fig, global_step=self.current_epoch)


logger = TensorBoardLogger(
    save_dir="lightning_logs",
    name="contrastive_model_logs"
)

batch_size = 64
X, X3D = generate_line_data(batch_size * 10, logger)
dataset = ContrastiveDataset(X, X3D)
shared_args = dict(
    batch_size=batch_size,
    num_workers=23,
    drop_last=True
)
train_loader = DataLoader(RepeatDataset(dataset, 10), shuffle=True, **shared_args)
val_loader = DataLoader(dataset, shuffle=False, **shared_args)

# Train Model
model = ContrastiveModel(
    input_dim=3,
    output_dim=2,
    lr=.5e-3,
    eq=False
)
trainer = pl.Trainer(
    devices=[1],
    max_epochs=20,
    check_val_every_n_epoch=1,
    log_every_n_steps=5,
    logger=logger)
trainer.fit(model, train_loader, val_loader)