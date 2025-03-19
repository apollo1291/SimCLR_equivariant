import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.cross_decomposition import CCA

def perform_cca(diff, shifts, n_components=1):
    """
    Performs CCA between diff (shape [N, d]) and shifts (shape [N, 1]).
    Returns the canonical correlation coefficient (between the first pair).
    """
    cca = CCA(n_components=n_components)
    # Fit and transform. Make sure shifts is reshaped as [N, 1].
    Y = shifts.reshape(-1, 1)
    cca.fit(diff, Y)
    X_c, Y_c = cca.transform(diff, Y)
    # Compute the correlation coefficient between the first canonical variates.
    corr_matrix = np.corrcoef(X_c[:,0], Y_c[:,0])
    corr = corr_matrix[0,1]
    return corr

def main():
    # Folder containing saved checkpoints.
    checkpoint_folder = "saved_runs"
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pt")))
    
    # Lists to store values over different alpha.
    alphas = []
    cca_full_list = []          # CCA correlation over the entire dataset.
    cca_class0_list = []        # CCA correlation for class 0.
    # (Optionally, one could also count the number of significant directions—for 1D shift this is trivial.)
    
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        alpha = ckpt["alpha"]
        # Load required keys:
        # "projected_points" (f(x)), "augmented_projected_points" (f(T(x))),
        # "shifts", and "ids"
        proj = ckpt["projected_points"]          # shape: [N, d]
        aug_proj = ckpt["augmented_projected_points"]  # shape: [N, d]
        shifts = ckpt["shifts"]                    # shape: [N] or [N, 1]
        ids = ckpt["ids"]                          # shape: [N]
        
        # Convert to numpy arrays.
        proj = proj.clone().detach().float().numpy()
        aug_proj = aug_proj.clone().detach().float().numpy()
        # Flatten shifts to shape [N]
        shifts = shifts.clone().detach().float().numpy().flatten()
        ids = ids.clone().detach().long().numpy()
        
        # Compute the representation difference (i.e. the effect of the transformation).
        diff = aug_proj - proj  # shape: [N, d]
        
        # Perform CCA on the entire dataset.
        corr_full = perform_cca(diff, shifts, n_components=1)
        
        # For class-specific analysis, select only samples where id == 0.
        class0_indices = np.where(ids == 0)[0]
        if len(class0_indices) > 0:
            diff_class0 = diff[class0_indices]
            shifts_class0 = shifts[class0_indices]
            corr_class0 = perform_cca(diff_class0, shifts_class0, n_components=1)
        else:
            corr_class0 = np.nan
        
        alphas.append(alpha)
        cca_full_list.append(corr_full)
        cca_class0_list.append(corr_class0)
        
        print(f"File: {ckpt_file} | alpha: {alpha:.2f} | CCA Corr (full): {corr_full:.3f} | CCA Corr (class0): {corr_class0:.3f}")
    
    alphas = np.array(alphas)
    cca_full_array = np.array(cca_full_list)
    cca_class0_array = np.array(cca_class0_list)
    
    # --- Plotting ---
    fig1, ax1 = plt.subplots()
    ax1.set_ylim(0.0, 1.1)
    ax1.plot(alphas, cca_full_array, marker="o", linestyle="-")
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Canonical Correlation (Full Dataset)")
    ax1.set_title("CCA Correlation vs. Alpha (Full Dataset)")
    
    fig2, ax2 = plt.subplots()
    ax2.set_ylim(0.0, 1.1)
    ax2.plot(alphas, cca_class0_array, marker="o", linestyle="-", color="purple")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Canonical Correlation (Class 0)")
    ax2.set_title("CCA Correlation vs. Alpha (Class 0)")
    
    writer = SummaryWriter(log_dir="lightning_logs/contrastive_model_logs/CCAAnalysis_Debug", comment="CCA Debug")
    writer.add_figure("CCA/Full_Dataset_vs_Alpha", fig1, global_step=0)
    writer.add_figure("CCA/Class0_vs_Alpha", fig2, global_step=0)
    writer.close()
    
    plt.show()

if __name__ == "__main__":
    main()
