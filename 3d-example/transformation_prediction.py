import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class TransformationPredictor(pl.LightningModule):
    def __init__(self, input_dim, lr):
        """
        A simple linear probe that maps the difference between the transformed and original 
        representations to a scalar prediction of the shift.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.lr = lr

    def forward(self, diff):
        return self.linear(diff)

    def training_step(self, batch, batch_idx):
        projected, augmented_projected, shift = batch
        diff = augmented_projected - projected  # difference representing the effect of transformation
        pred = self(diff).squeeze(1)  # shape: [batch]
        loss = F.mse_loss(pred, shift)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        projected, augmented_projected, shift = batch
        diff = augmented_projected - projected
        pred = self(diff).squeeze(1)
        loss = F.mse_loss(pred, shift)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# --- Dataset Wrapper for a Checkpoint ---

class CheckpointDataset(Dataset):
    def __init__(self, projected_points, augmented_projected_points, shifts):
        """
        Expects tensors for:
          - projected_points: f(x) of shape [N, d]
          - augmented_projected_points: f(T(x)) of shape [N, d]
          - shifts: transformation parameter, shape [N]
        """
        self.projected_points = projected_points
        self.augmented_projected_points = augmented_projected_points
        self.shifts = shifts

    def __len__(self):
        return self.projected_points.shape[0]

    def __getitem__(self, idx):
        return (self.projected_points[idx],
                self.augmented_projected_points[idx],
                self.shifts[idx])

# --- Main Experiment ---

def main():
    # Folder containing saved checkpoints.
    checkpoint_folder = "saved_runs/epoch_100"
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pt")))
    
    # Lists to store alpha values and the corresponding validation performance (MSE)
    alphas = []
    val_mse_list = []
    
    # For each checkpoint (each corresponding to a different alpha)
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        alpha = ckpt["alpha"]
        projected_points = ckpt["projected_points"]
        augmented_projected_points = ckpt["augmented_projected_points"]
        shifts = ckpt["shifts"]

        # Ensure these are torch.FloatTensors (and shifts as float, squeezed to shape [N])
        if not torch.is_tensor(projected_points):
            projected_points = torch.tensor(projected_points)
        else:
            projected_points = projected_points.clone().detach().float()
        if not torch.is_tensor(augmented_projected_points):
            augmented_projected_points = torch.tensor(augmented_projected_points)
        else:
            augmented_projected_points = augmented_projected_points.clone().detach().float()
        if not torch.is_tensor(shifts):
            shifts = torch.tensor(shifts)
        else:
            shifts = shifts.clone().detach().float().squeeze()

        # Create dataset and dataloaders
        dataset = CheckpointDataset(projected_points, augmented_projected_points, shifts)
        n_total = len(dataset)
        n_val = int(0.2 * n_total)
        n_train = n_total - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        # Create and train the TransformationPredictor model.
        # Input dimension is the difference dimension (which equals the dimension of the representations).
        input_dim = projected_points.shape[1]
        pl.seed_everything(0)
        model = TransformationPredictor(input_dim=input_dim, lr=1e-3)

        # Create a PyTorch Lightning Trainer
        trainer = pl.Trainer(
            max_epochs=100,
            logger=False,  # We log later globally via tensorboard_summary below
            enable_checkpointing=False,
            devices=[0],
        )

        trainer.fit(model, train_loader, val_loader)

        # After training, get the final validation loss
        val_result = trainer.callback_metrics.get("val_loss")
        val_mse = val_result.item() if val_result is not None else np.nan
        alphas.append(alpha)
        val_mse_list.append(val_mse)
        print(f"Checkpoint {ckpt_file} (alpha = {alpha:.2f}): Validation MSE = {val_mse:.4f}")

    # --- Plotting ---
    alphas = np.array(alphas)
    val_mse_array = np.array(val_mse_list)

    fig, ax = plt.subplots()
    ax.plot(alphas, val_mse_array, marker="o", linestyle="-")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Validation MSE (Shift Prediction)")
    ax.set_title("Transformation Predictability vs. Alpha")

    # --- Log Figure to TensorBoard ---
    writer = SummaryWriter(log_dir="lightning_logs/contrastive_model_logs/TransPredict")
    writer.add_figure("TransPredict/Val_MSE_vs_Alpha", fig, global_step=0)
    writer.close()
    plt.show()

if __name__ == "__main__":
    main()
