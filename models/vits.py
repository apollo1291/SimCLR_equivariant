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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ChainedScheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm

import timm

from utils import InfoNCE, accuracy, contrast_loss, RollingAvg
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
    features: torch.Tensor
    
class BaseKQConModel(pl.LightningModule):
    def __init__(self, model, mlp_dim=1024, args=None) -> None:
        #.to(args.device)
        #self.device = device
        #print(model.device)
       #self.fourier_encoder_fn = fourier_encoder
        super().__init__()

        self.num_bands = 6
        self.num_transfrom_params = 7
        self.encoding_size = self.num_bands * self.num_transfrom_params * 2

        self.model = model

        self._build_projector_and_predictor_mlps(model.vit.embed_dim, mlp_dim)


        
        self.loss_fn = contrast_loss # InfoNCE()
        self.linear_classifier_loss = nn.CrossEntropyLoss()

        self.args = args

        self.save_hyperparameters(ignore=['model'])
        self.automatic_optimization = False

        self.checkpoint_dict = {}

        self.collapse_ck = False
        self.grad_avg = RollingAvg(50, nonzero=True)
        self.adaptive_clipping = True
        #self.grad_clipping
        
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

        #return x

    def _forward(self, x1, x2, t1, t2=None, use_fourier=True):

        device = x1.device
        torch.save(x1, "checkpoints/images1.pth")
        t1 = t1.to(device)

        CLSq1, CLSq2 = None, None
        if use_fourier:

            z1, predicted_z2 = self.forward(x3d, shift, eq=True)
            z2, predicted_z1 = self.forward(augmented_x3d, -shift, eq=True)
            a = self.info_nce_loss(z1, predicted_z1)
            b = self.info_nce_loss(predicted_z2, z2)
            c = alpha * torch.linalg.vector_norm(z1 - z2)
            loss = a + b - c
            self.log("train_loss", loss)

            fea1 = self.fourier_encoder_fn(t1)

            CLSq1 = self.projector(fea1)


        image_rep1, predicted_rep2 = self.model(x1, CLSq1)
        image_rep2, predicted_rep1 = self.model(x2, CLSq2)
        torch.save(image_rep1, "checkpoints/rep1_model_outputs.pth")
        torch.save(image_rep2, "checkpoints/rep2_model_outputs.pth")
        

        """
        Curently, only img_loss2 is valid because, predicted_rep1 is not well defined
        as a result of the asymetrical fourier encoding parameters, CLSq2 is always None
        """
        if use_fourier:
            #img1_loss, img1_logits, img1_labels = self.loss_fn(image_rep1, predicted_rep1)
            sims = torch.einsum("bc,dc->bd", image_rep2, predicted_rep2)
            img2_loss, img2_logits, img2_labels = self.loss_fn(sims, loss_leak=self.args.loss_leak)
            
            loss =  img2_loss # + img1_loss
            logits = img2_logits
            labels = img2_labels
        else:
            #TODO: look into this, we want diagonal to be large positive off diagonal should all be small negative 
            
            sims = torch.einsum("bc,dc->bd", image_rep1,  image_rep2)
            loss, logits, labels = self.loss_fn(sims, loss_leak=self.args.loss_leak)

        return ForwardOutput(
            loss=loss, 
            image_rep1=image_rep1, 
            predicted_rep2=predicted_rep2, 
            image_rep2=image_rep2, 
            predicted_rep1=predicted_rep1, 
            logits=logits, 
            labels=labels, 
            features=sims
        )

    def training_step(self, batch, batch_idx):
        
        optimizer_embedding, optimizer_classifier = self.optimizers()
        
        images, params, class_ids = batch
        x1, x2 = images[0], images[1]
        x1_to_x2_params = transformation_params_to_tensor_batch(params[0])
        #t2 = transformation_params_to_tensor_batch(params[1])

        
        forward_output = self.forward(x1, x2, x1_to_x2_params, use_fourier=self.args.use_fourier)
        contrastive_loss, logits, labels = forward_output.loss, forward_output.logits, forward_output.labels

        # self.on_before_optimizer_step(optimizer_embedding)
        optimizer_embedding.zero_grad()
        self.manual_backward(contrastive_loss)
        optimizer_embedding.step()

        # Detach embeddings and compute classification loss for the evaluation head

        image_rep1, image_rep2 = torch.clone(forward_output.image_rep1.detach()), torch.clone(forward_output.image_rep2.detach()) #TODO: try this: detached_code = torch.clone(code.detach())
        eval_logits1, eval_logits2 = self.linear_classifier(image_rep1), self.linear_classifier(image_rep2)
        #print(class_ids)
        class_ids = torch.tensor(class_ids).to(eval_logits1.device)
        classification_loss = self.linear_classifier_loss(eval_logits1, class_ids) + self.linear_classifier_loss(eval_logits2, class_ids)

        optimizer_classifier.zero_grad()
        self.manual_backward(classification_loss)
        optimizer_classifier.step()


        if batch_idx % 10 == 0:
            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            self.log('train_loss', contrastive_loss)
            self.log('train_acc_top1', top1[0])
            self.log('train_acc_top5', top5[0])
            lr = optimizer_embedding.param_groups[0]['lr']
            self.log('lr', lr, on_step=True, on_epoch=False)
            
            top1_1, top5_1 = accuracy(eval_logits1, class_ids, topk=(1, 5))
            top1_2, top5_2 = accuracy(eval_logits2, class_ids, topk=(1, 5))

       
            self.log('train_class_acc_top1', (top1_1[0] + top1_2[0]) / 2, prog_bar=True, on_epoch=False, batch_size=len(class_ids), sync_dist=True)
            self.log('train_class_acc_top5', (top5_1[0] + top5_2[0]) / 2, prog_bar=True, on_epoch=False, batch_size=len(class_ids), sync_dist=True)

            self.log('linear_class_train_loss', classification_loss, prog_bar=True)

        

        if torch.isnan(classification_loss) and not self.collapse_ck:
            # Save the current state of the linear classifier using PL's save_checkpoint
            self.checkpoint_dict["collapse_batch"] = {
                'images': [images[0].cpu(), images[1].cpu()],
                #'params': params,
                #'class_ids': class_ids,
                'features': forward_output.features.cpu(),
                'epoch': self.current_epoch,
                'batch_idx': batch_idx
                }
            
            torch.save(self.checkpoint_dict, f'checkpoints/{self.args.name}_batch_checkpoint_epoch_{self.current_epoch}_batch_{batch_idx}.pth')
                
            self.trainer.save_checkpoint(
            f"checkpoints/{self.args.name}_model_checkpoint_epoch_{self.current_epoch}_batch_{batch_idx}.ckpt"
        )
            self.collapse_ck = True

        if batch_idx % 5 == 0:
            self.checkpoint_dict["batch_5"] = {
                'images': [images[0].cpu(), images[1].cpu()],
                #'params': [params[0].cpu(), params[1].cpu()],
                #'class_ids': class_ids.cpu(),
                'features': forward_output.features.cpu(),
                'epoch': self.current_epoch,
                'batch_idx': batch_idx
            }
        # Save the last batch
        self.checkpoint_dict['previous_batch'] = {
            'images': [images[0].cpu(), images[1].cpu()],
            #'params': [params[0].cpu(), params[1].cpu()],
            #'class_ids': class_ids.cpu(),
            'features': forward_output.features.cpu(),
            'epoch': self.current_epoch,
            'batch_idx': batch_idx
        }

        scheduler = self.lr_schedulers()
        scheduler.step()

        return contrastive_loss
        

    
    def validation_step(self, batch, batch_idx):
        images, params, class_ids = batch
        x1, x2 = images[0], images[1]
        

        # Generate representations using frozen embedding model
        with torch.no_grad():
            x1_to_x2_params = transformation_params_to_tensor_batch(params[0])
            forward_output = self.forward(x1, x2, x1_to_x2_params, use_fourier=self.args.use_fourier)
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
        embedding_optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )

        self.linear_classifier_optimizer = torch.optim.Adam(
            self.linear_classifier.parameters(),
            lr=self.args.lr * 10,
            weight_decay=self.args.weight_decay
        )

        warmup_steps = self.total_steps * 0.2  
        total_steps = self.total_steps

        warmup_scheduler = LinearLR(
            optimizer=embedding_optimizer,
            start_factor=0.01,   
            total_iters=warmup_steps   
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer=embedding_optimizer,
            T_max=total_steps - warmup_steps,  
            eta_min=0                            
        )
        scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

        
        return (
            [embedding_optimizer, self.linear_classifier_optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            ]
        )

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self, norm_type=2)
        avg_grads = self.grad_avg.get_all()
        params = {
            f"grad_2.0_norm/{name}": p
            for name, p in self.named_parameters()
            if p.grad is not None
        }

        if self.adaptive_clipping:
            for k in norms.keys():
                if k in params:
                    avg_grad = max(avg_grads.get(k, norms[k]), 1e-5)
                    if norms[k] > avg_grad * 5 and self.global_step > 0:
                        print(f"Bad grad for {k}: {norms[k]} scaling to {avg_grad * 5}")
                        torch.nn.utils.clip_grad_norm_(params[k], avg_grad * 5)
                        norms[k] = avg_grad * 5

                    # if norms[k] > self.gradient_clipping:
                    #     # print(f"Bad grad for {k}: {norms[k]} scaling to {self.gradient_clipping}")
                    #     torch.nn.utils.clip_grad_norm_(params[k], self.gradient_clipping)

        # self.grad_avg.add_all(norms)
        # self.log_dict(norms)

class KQConModel(BaseKQConModel):
    def __init__(self, model,  num_steps, params_size, mlp_dim=1024, args=None,):
        #self.device = model.device
        self.param_size = params_size
        super().__init__(model=model, mlp_dim=mlp_dim, args=args,)
        self.total_steps = num_steps * self.args.epochs
    
    def get_feature_encoding_size(self):
        return self.encoding_size

    def _build_projector_and_predictor_mlps(self, embed_dim, mlp_dim, num_layers=10, output_dim=1000):
        input_dim = self.get_feature_encoding_size()
        self.projector = self._build_mlp(num_layers, input_dim=input_dim, mlp_dim=mlp_dim, output_dim=embed_dim, last_bn=True)#.to(self.device)

        #TODO: simplify to single layer, Use layer norm on channel dim
        self.linear_classifier = self._build_mlp(3, self.model.rep_size, 128, output_dim=NUM_CLASS, last_bn=False)#.to(self.device)
    
    def forward(self, x1, x2, t1, t2=None, use_fourier=False):
        output = self._forward(x1, x2, t1, use_fourier)
        return output





class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', rep_size=128):
        super(ViT, self).__init__()

        self.rep_size = rep_size

        
        self.vit = timm.create_model(model_name, pretrained=False) # maybe switch 
        

        # Create two learnable class tokens
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))

        
        nn.init.normal_(self.cls_token1, std=0.02)
        nn.init.normal_(self.cls_token2, std=0.02)

        # Disable the ViT's classifier head 
        self.vit.head = nn.Identity()
        
        # Define a custom linear head for the image representation
        self.image_head = nn.Sequential(
            nn.Linear(self.vit.embed_dim, self.vit.embed_dim), 
            nn.ReLU(), 
            nn.Linear(self.vit.embed_dim, rep_size)
        )
        
        # Define a custom linear head for the additional input
        self.additional_head = nn.Sequential(
            nn.Linear(self.vit.embed_dim, self.vit.embed_dim), 
            nn.ReLU(), 
            nn.Linear(self.vit.embed_dim, rep_size)
        )
        
    def forward(self, img, fourier_encoding=None):

        img_embeddings = self.vit.patch_embed(img)

        batch_size = img_embeddings.shape[0]

        img_embeddings = torch.cat((img_embeddings, self.cls_token1.expand(batch_size, -1, -1)), dim=1)

        first_token  = self.cls_token1.expand(batch_size, -1, -1)
        second_token = self.cls_token2.expand(batch_size, -1, -1)

        if fourier_encoding is not None:
            second_token = second_token + fourier_encoding.unsqueeze(1)

        augmented_embeddings = torch.cat((img_embeddings, first_token, second_token), dim=1)

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
        self.vit.pos_drop(augmented_embeddings)
        
        # Pass the augmented sequence through the Transformer blocks
        x = self.vit.blocks(augmented_embeddings)
        x = self.vit.norm(x)
        
        # Extract the representations for the image and the additional input
        image_representation = x[:, -2, :]  # The second to last token is the class token with no FE, 
        pair_representation = x[:, -1, :]   # The last token is includes FE information
        
        image_output = self.image_head(image_representation)  
        pair_output = self.additional_head(pair_representation)
        
        return image_output, pair_output


class ViTModel(pl.LightningModule):
    def __init__(self, vit_model, num_steps, args=None):
        super().__init__()
        self.vit = vit_model  # Use the provided ViT model
        self.classifier = nn.Linear(self.vit.rep_size, NUM_CLASS)  # Adjust the head for the number of classes
        self.criterion = nn.CrossEntropyLoss()  # Define the loss function

    def forward(self, x):
        x, _ = self.vit(x)  
        return self.classifier(x)


    def training_step(self, batch, batch_idx):
        x, _, y = batch  
        img1, img2 = x
        logits1, logits2 = self(img1), self(img2)  # Forward pass
        class_ids = torch.tensor(y).to(logits1.device)
        loss = self.criterion(logits1, class_ids) + self.criterion(logits2, class_ids)  # Compute the loss
        acc = (accuracy(logits1, class_ids,  topk=(1, ))[0] + accuracy(logits2, class_ids,  topk=(1,))[0]) / 2  # Compute the loss
        self.log('train_loss', loss)  # Log the training loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, _, y = batch  
        img1, img2 = x
        logits1, logits2 = self(img1), self(img2)  # Forward pass
        class_ids = torch.tensor(y).to(logits1.device)
        loss = self.criterion(logits1, class_ids) + self.criterion(logits2, class_ids)  # Compute the loss
        acc = (accuracy(logits1, class_ids,  topk=(1, ))[0] + accuracy(logits2, class_ids,  topk=(1,))[0]) / 2  # Compute accuracy using the previously defined accuracy function
        self.log('val_loss', loss)  # Log the validation loss
        self.log('val_accuracy', acc[0])  # Log the validation accuracy

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)  # Define the optimizer
        return optimizer