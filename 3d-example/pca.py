import os
import torch
import glob
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

def run_pca(data, n_components=2):
    """
    Runs PCA on the given data using SVD.
    
    Args:
        data (Tensor): shape [N, d] data.
        n_components (int): number of principal components to return.
        
    Returns:
        mean (Tensor): Mean of the data.
        components (Tensor): Principal components as columns (shape [d, n_components]).
        explained_variance (Tensor): Explained variances for the principal components.
    """
    # Center the data
    mean = data.mean(dim=0, keepdim=True)
    data_centered = data - mean
    # Compute SVD
    U, S, V = torch.linalg.svd(data_centered, full_matrices=False)
    # Principal components (rows of V; we want them as columns)
    components = V.T[:, :n_components]  # shape [d, n_components]
    # Compute explained variance: eigenvalues of covariance matrix
    # Note: S contains singular values, so variance = S^2/(N-1)
    explained_variance = (S[:n_components] ** 2) / (data.shape[0] - 1)
    return mean, components, explained_variance

def plot_pca(data, mean, components, explained_variance, scale_factor=3.0):
    """
    Plot the data along with the principal component directions.
    
    Args:
        data (Tensor): shape [N, d]
        mean (Tensor): mean of the data (1, d)
        components (Tensor): principal components (d, n_components)
        explained_variance (Tensor): explained variance of each component.
        scale_factor (float): scaling factor for arrow lengths.
        
    Returns:
        fig: the matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    
    
    ax.scatter(data[:, 0].numpy(), data[:, 1].numpy(), alpha=0.6, label='Projected Points')
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title("PCA on Projected Points")
    
    # Plot the principal component directions as arrows
    center = mean.squeeze().numpy()  # center point
    for i in range(components.shape[1]):
        # Scale arrow length by sqrt of explained variance times a scaling factor
        comp = components[:, i]
        length = scale_factor * torch.sqrt(explained_variance[i]).item()
        arrow = comp * length
        ax.arrow(center[0], center[1],
                 arrow[0], arrow[1],
                 head_width=0.1, head_length=0.1, fc='r', ec='r',
                 linewidth=2, label=f"PC{i+1}" if i == 0 else None)
    
    ax.legend()
    return fig

def main():
    checkpoint_files = sorted(glob.glob(os.path.join("saved_runs", "*.pt")))


    writer = SummaryWriter(log_dir="lightning_logs/contrastive_model_logs/pcaAnalysis")
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        
    
        projected_points = ckpt["projected_points"]  # shape [N, d] (d=2)
        projected_points = projected_points.to(torch.float32)
        
        # Run PCA on the projected points
        mean, components, explained_variance = run_pca(projected_points, n_components=2)
        print("Mean:", mean)
        print("Principal Components:\n", components)
        print("Explained Variance:", explained_variance)
        
        fig = plot_pca(projected_points, mean, components, explained_variance)
        
        writer.add_figure(f"PCA/ProjectedPoints_{ckpt_file}", fig, global_step=0)
    writer.close()
    

if __name__ == "__main__":
    main()
