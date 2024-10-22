import os
import sys
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.models.vision_transformer import VisionTransformer, _cfg
from torchvision import transforms

from info_nce import InfoNCE, info_nce
from utils import InfoNCE, info_nce, save_config_file, accuracy, save_checkpoint
from data_aug.contrastive_learning_dataset import transformation_params_to_tensor_batch

NUM_CLASS = 10

@dataclass
class ForwardOutput:
    loss: torch.Tensor
    image_rep1: torch.Tensor
    predicted_rep2: torch.Tensor
    image_rep2: torch.Tensor
    predicted_rep1: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    
class BaseKQConModel(nn.Module):
    def __init__(self, model, dim=256, mlp_dim=4096, args=None, optimizer=None, scheduler=None) -> None:
        super().__init__()
        self.model = model.to(args.device)
       #self.fourier_encoder_fn = fourier_encoder

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        self.loss_fn =  InfoNCE()

        self.args = args
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter(log_dir="/runs/simclr")
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
    
    # From mocoV3
    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
    
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass
    
    def fourier_encoder_fn(self, x, num_bands=6, max_freq=10.0):
        """
        Apply Fourier feature mapping to input tensor x.
        """
        batch_size, num_params = x.shape
        freq_bands = torch.linspace(1.0, max_freq, num_bands).to(x.device)
        x_expanded = x.unsqueeze(-1)  # Shape: (batch_size, num_params, 1)
        freq_bands = freq_bands.view(1, 1, -1)  # Shape: (1, 1, num_bands)
        x_freq = x_expanded * freq_bands * 2 * np.pi  # Shape: (batch_size, num_params, num_bands)
        sin_x = torch.sin(x_freq)
        cos_x = torch.cos(x_freq)
        pe = torch.cat([sin_x, cos_x], dim=-1)  # Shape: (batch_size, num_params, num_bands * 2)
        pe = pe.view(batch_size, -1)  # Flatten to (batch_size, num_params * num_bands * 2)
        return pe

    def _forward(self, x1, x2, t1, t2, use_fourier=True):
        t_diff = t2 - t1

        CLSq1, CLSq2 = None, None
        if use_fourier:
            fea1, fea2 = self.fourier_encoder_fn(t_diff), self.fourier_encoder_fn(-t_diff)
            CLSq1, CLSq2 = self.projector(fea1), self.projector(fea2)

        image_rep1, predicted_rep2 = self.model(x1, CLSq1)
        image_rep2, predicted_rep1 = self.model(x2, CLSq2)

        if use_fourier:
            img1_loss, img1_logits, img1_labels = self.loss_fn(image_rep1, predicted_rep1)
            img2_loss, img2_logits, img2_labels = self.loss_fn(image_rep2, predicted_rep2)
            loss = img1_loss + img2_loss
            logits = torch.cat([img1_logits, img2_logits], dim=0)
            labels = torch.cat([img1_labels, img2_labels], dim=0)
        else:
            loss, logits, labels = self.loss_fn(image_rep1, image_rep2)

        return ForwardOutput(loss, image_rep1, predicted_rep2, image_rep2, predicted_rep1, logits, labels)

    def train(self, train_loader, vol, use_fourier=False):

        scaler = GradScaler(enabled=self.args.fp16_precision)
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, params in tqdm(train_loader):
                x1, x2 = images[0].to(self.args.device), images[1].to(self.args.device)
                t1, t2 = transformation_params_to_tensor_batch(params[0]).to(self.args.device), transformation_params_to_tensor_batch(params[1]).to(self.args.device)  # Assign transformation parameters if applicable

                with autocast(enabled=self.args.fp16_precision):
                    # Forward pass through the model
                    output = self.forward(x1, x2, t1, t2, use_fourier)
                    loss, logits, labels = output.loss, output.logits, output.labels

                    self.optimizer.zero_grad()

                    scaler.scale(loss).backward()

                    scaler.step(self.optimizer)
                    scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                    vol.commit()

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        
        vol.commit()

class KQConModel(BaseKQConModel):
    def __init__(self, model, dim=256, mlp_dim=4096, args=None, optimizer=None, scheduler=None):
        super().__init__(model=model, dim=dim, mlp_dim=mlp_dim, args=args, optimizer=optimizer, scheduler=scheduler)
    
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, num_layers=3, output_dim=100):
        self.predictor = self._build_mlp(num_layers, dim, mlp_dim, output_dim=self.model.vit.embed_dim, last_bn=True)
        self.linear_classifier = self._build_mlp(num_layers, self.model.rep_size, mlp_dim, output_dim=NUM_CLASS, last_bn=False)
    
    def forward(self, x1, x2, t1, t2, use_fourier):
        output = self._forward(x1, x2, t1, t2, use_fourier)
        return output




class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', rep_size=1000):
        super(ViT, self).__init__()

        self.rep_size = rep_size

        self.vit = timm.create_model(model_name, pretrained=True)
        
        # Disable the ViT's classifier head 
        self.vit.head = nn.Identity()
        
        # Define a custom linear head for the image representation
        self.image_head = nn.Linear(self.vit.embed_dim, rep_size)
        
        # Define a custom linear head for the additional input
        self.additional_head = nn.Linear(self.vit.embed_dim, rep_size)
        
    def forward(self, img, fourier_encoding):

        img_embeddings = self.vit.patch_embed(img)

        if fourier_encoding is not None:
            # Incorporate the Fourier encoding if provided
            fourier_encoding = fourier_encoding.unsqueeze(1)  # Shape: (batch_size, 1, embed_dim)
            augmented_embeddings = torch.cat((img_embeddings, fourier_encoding), dim=1)
        else:
            augmented_embeddings = img_embeddings
        # Add positional embeddings (if needed)
        if self.vit.pos_embed.shape[1] < augmented_embeddings.shape[1]:
            # Interpolate position embeddings to match the length
            pos_embed = torch.nn.functional.interpolate(
                self.vit.pos_embed.transpose(1, 2),
                size=(augmented_embeddings.shape[1]),
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        else:
            pos_embed = self.vit.pos_embed[:, :augmented_embeddings.shape[1], :]
        
        # Add positional embeddings to the sequence
        augmented_embeddings += pos_embed
        
        # Pass the augmented sequence through the Transformer blocks
        x = self.vit.blocks(augmented_embeddings)
        x = self.vit.norm(x)
        
        # Extract the representations for the image and the additional input
        image_representation = x[:, :-1, :]  # All but the last token are image patches
        pair_representation = x[:, -1, :]   # The last token is the additional embedding
        
        # Compute separate outputs for image and additional input
        image_output = self.image_head(image_representation.mean(dim=1))  # Pool the image tokens
        pair_output = self.additional_head(pair_representation)         # Single token for extra input
        
        return image_output, pair_output