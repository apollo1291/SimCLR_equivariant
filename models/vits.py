import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer, _cfg
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from info_nce import InfoNCE, info_nce
import numpy as np


class KQConModel(nn.Module):
    def __init__(self, model, dim=256, mlp_dim=4096) -> None:
        super(KQConModel, self).__init__()

        self.model = model
       #self.fourier_encoder_fn = fourier_encoder

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        self.loss_fn = InfoNCE()
    
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


    def forward(self, x1, x2, t1, t2):
        t_diff = t2 - t1  # t1 and t2 are tensors of transformation parameters

        fea1 = self.fourier_encoder_fn(t_diff)
        fea2 = self.fourier_encoder_fn(-t_diff)

        CLSq1 = self.projector(fea1)
        CLSq2 = self.projector(fea2)

        # Pass through the model
        image_rep1, predicted_rep2 = self.model(x1, CLSq1)
        image_rep2, predicted_rep1 = self.model(x2, CLSq2)

        return self.loss_fn(image_rep1, predicted_rep1) + self.loss_fn(image_rep2, predicted_rep2)


class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', rep_size=1000):
        super(ViT, self).__init__()
    
        self.vit = timm.create_model(model_name, pretrained=True)
        
        # Disable the ViT's classifier head 
        self.vit.head = nn.Identity()
        
        # Define a custom linear head for the image representation
        self.image_head = nn.Linear(self.vit.embed_dim, rep_size)
        
        # Define a custom linear head for the additional input
        self.additional_head = nn.Linear(self.vit.embed_dim, rep_size)
        
    def forward(self, img, fouirer_encoding):

        img_embeddings = self.vit.patch_embed(img)
        # Concatenate the additional embedding with the image embeddings
        augmented_embeddings = torch.cat((img_embeddings, fouirer_encoding), dim=1)  # Shape: (batch_size, num_patches+1, embed_dim)
        
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