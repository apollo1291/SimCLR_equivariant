import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import glob
import os 


class DownstreamDataset(Dataset):

    def __init__(self, representations, ids):
        self.representations = representations
        self.labels = ids 

    def __len__(self):
        return len(self.representations)
    
    def __getitem__(self, idx):
        return self.representations[idx], self.labels[idx]

class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.classifer = nn.Sequential(nn.Linear(2, 5), nn.ReLU(), nn.Linear(5, 5), nn.ReLU(), nn.Linear(5, 5))
        self.lr = 0.01
        self.loss = torch.nn.CrossEntropyLoss()


    def training_step(self, batch, batch_idx):

        points, labels = batch
        preds = self.classifer(points)
        loss = self.loss(preds, labels)

        self.log("class_train_loss", loss)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        points, labels = batch
        preds = self.classifer(points)
        loss = self.loss(preds, labels)

        _, predicted = torch.max(preds, 1)
        acc = (predicted == labels).sum().item() / len(labels)

        self.log("val_classifier_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc,  on_step=False, on_epoch=True, prog_bar=True)

        return loss

    

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)



if __name__ == "__main__":
    # Folder containing saved checkpoints.
    checkpoint_folder = "saved_runs/inverse/shift"
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pt")))
    
    
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        
        # Each checkpoint is assumed to contain:
        # "alpha", "projected_points", "shifts", and "ids"
        alpha = ckpt["alpha"]
        representations = ckpt["projected_points"]  # shape: [N, d] (e.g. d=2)
        shifts = ckpt["shifts"]                     # shape: [N] or [N, 1]
        ids = ckpt["ids"]                           # shape: [N]



        shared_args = dict(
        batch_size=64,
        num_workers=0,  # adjust as needed
        drop_last=True
        )
        train_representations, val_representations, train_ids, val_ids = train_test_split(representations, ids, test_size=0.4, random_state=42)
        train_dataset = DownstreamDataset(train_representations, train_ids)
        val_dataset = DownstreamDataset(val_representations, val_ids)

        train_loader = DataLoader(train_dataset, shuffle=True, **shared_args)
        val_loader = DataLoader(val_dataset, shuffle=False, **shared_args)

        model = Model()

        logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name="classifier",
        version=f"classifier_{alpha:.2f}"
        )
        
        trainer = pl.Trainer(
            devices=[0],  # or [1], depending on your GPU setup
            max_epochs=20,  # reduce or increase as desired
            check_val_every_n_epoch=1,
            log_every_n_steps=1,
            logger=logger
        )

        trainer.fit(model, train_loader, val_loader)


        