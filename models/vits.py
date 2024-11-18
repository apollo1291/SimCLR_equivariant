import os
import sys
import logging
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
import pytorch_lightning as pl

import timm

from utils import InfoNCE, accuracy
from data_aug.contrastive_learning_dataset import transformation_params_to_tensor_batch

import pytorch_lightning as pl

NUM_CLASS = 1000

@dataclass
class ForwardOutput:
    loss: torch.Tensor
    image_rep1: torch.Tensor
    predicted_rep2: torch.Tensor
    image_rep2: torch.Tensor
    predicted_rep1: torch.Tensor
    logits: torch.Tensor
    labels: torch.Tensor
    
class BaseKQConModel(pl.LightningModule):
    def __init__(self, model, mlp_dim=1024, args=None,) -> None:
        #.to(args.device)
        #self.device = device
        #print(model.device)
       #self.fourier_encoder_fn = fourier_encoder
        super().__init__()

        self.num_bands = 6
        self.num_transfrom_params = 12
        self.encoding_size = self.num_bands * self.num_transfrom_params * 2

        self.model = model

        self._build_projector_and_predictor_mlps(model.vit.embed_dim, mlp_dim)


        
        self.loss_fn =  InfoNCE()
        self.linear_classifier_loss = nn.CrossEntropyLoss()

        self.args = args

        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False
        
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
    
    def fourier_encoder_fn(self, x,  max_freq=10.0):
        """
        Apply Fourier feature mapping to input tensor x.
        """
        batch_size, num_params = x.shape
        num_bands = self.num_bands
        assert num_params == self.num_transfrom_params
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

        device = x1.device
        t1, t2 = t1.to(device), t2.to(device)

        CLSq1, CLSq2 = None, None
        if use_fourier:
            fea1 = self.fourier_encoder_fn(t1)

            #print(fea1.device)
            #print(self.projector.device)
            CLSq1 = self.projector(fea1)

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

    def training_step(self, batch, batch_idx):
        
        optimizer_embedding, optimizer_classifier = self.optimizers()
        
        images, params, class_ids = batch
        x1, x2 = images[0], images[1]
        t1 = transformation_params_to_tensor_batch(params[0])
        t2 = transformation_params_to_tensor_batch(params[1])

        
        forward_output = self.forward(x1, x2, t1, t2, use_fourier=self.args.use_fourier)
        contrastive_loss, logits, labels = forward_output.loss, forward_output.logits, forward_output.labels

        
        if batch_idx % 10 == 0:
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.log('train_loss', contrastive_loss)
            self.log('train_acc_top1', top1[0])
            self.log('train_acc_top5', top5[0])
            lr = optimizer_embedding.param_groups[0]['lr']
            self.log('lr', lr, on_step=True, on_epoch=False)

        optimizer_embedding.zero_grad()
        self.manual_backward(contrastive_loss)
        optimizer_embedding.step()

        # Detach embeddings and compute classification loss for the evaluation head
        image_rep1, image_rep2 = forward_output.image_rep1.detach(), forward_output.image_rep2.detach()
        eval_logits1, eval_logits2 = self.linear_classifier(image_rep1), self.linear_classifier(image_rep2)
        #print(class_ids)
        class_ids = torch.tensor(class_ids).to(eval_logits1.device)
        classification_loss = self.linear_classifier_loss(eval_logits1, class_ids) + self.linear_classifier_loss(eval_logits2, class_ids)

        
        self.log('linear_class_train_loss', classification_loss, prog_bar=True)

        
        optimizer_classifier.zero_grad()
        self.manual_backward(classification_loss)
        optimizer_classifier.step()

        
        return contrastive_loss
    
    def validation_step(self, batch, batch_idx):
        images, params, class_ids = batch
        x1, x2 = images[0], images[1]
        

        # Generate representations using frozen embedding model
        with torch.no_grad():
            t1 = transformation_params_to_tensor_batch(params[0])
            t2 = transformation_params_to_tensor_batch(params[1])
            forward_output = self.forward(x1, x2, t1, t2, use_fourier=self.args.use_fourier)
            image_rep1, image_rep2 = forward_output.image_rep1, forward_output.image_rep2

        
            eval_logits1, eval_logits2 = self.linear_classifier(image_rep1), self.linear_classifier(image_rep2)
            #print(class_ids)
            class_ids = torch.tensor(class_ids).to(eval_logits1.device)
            classification_loss = self.linear_classifier_loss(eval_logits1, class_ids) + self.linear_classifier_loss(eval_logits2, class_ids)
        
        
        self.log('val_loss', classification_loss, prog_bar=True, on_epoch=True, batch_size=len(class_ids), sync_dist=True)

        
        top1_1, top5_1 = accuracy(eval_logits1, class_ids, topk=(1, 5))
        top1_2, top5_2 = accuracy(eval_logits2, class_ids, topk=(1, 5))

       
        self.log('val_acc_top1', (top1_1[0] + top1_2[0]) / 2, prog_bar=True, on_epoch=True, batch_size=len(class_ids), sync_dist=True)
        self.log('val_acc_top5', (top5_1[0] + top5_2[0]) / 2, prog_bar=True, on_epoch=True, batch_size=len(class_ids), sync_dist=True)

        return classification_loss


    
    def configure_optimizers(self):
        embedding_optimizer = torch.optim.Adam(self.parameters(), self.args.lr, weight_decay=self.args.weight_decay)

        self.linear_classifier_optimizer = torch.optim.Adam(
        self.linear_classifier.parameters(), lr=self.args.lr*10, weight_decay=self.args.weight_decay
    )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(embedding_optimizer, T_max=self.args.epochs, eta_min=0)
        return [embedding_optimizer, self.linear_classifier_optimizer], [{'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}]

class KQConModel(BaseKQConModel):
    def __init__(self, model,  mlp_dim=1024, args=None,):
        #self.device = model.device
        super().__init__(model=model, mlp_dim=mlp_dim, args=args,)
    
    def get_feature_encoding_size(self):
        return self.encoding_size

    def _build_projector_and_predictor_mlps(self, embed_dim, mlp_dim, num_layers=10, output_dim=1000):
        input_dim = self.get_feature_encoding_size()
        self.projector = self._build_mlp(num_layers,input_dim=input_dim, mlp_dim=mlp_dim, output_dim=embed_dim, last_bn=True)#.to(self.device)
        self.linear_classifier = self._build_mlp(num_layers, self.model.rep_size, mlp_dim, output_dim=NUM_CLASS, last_bn=False)#.to(self.device)
    
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
