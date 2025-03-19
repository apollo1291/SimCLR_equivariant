import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from scipy.stats import entropy

# --- Utility Functions for MI ---

def compute_mi_regression(features, target):
    """
    Computes the mutual information between a set of continuous features
    (features: shape [N, d]) and a continuous target (shape [N]).
    Returns the average MI across feature dimensions.
    """
    mi_values = mutual_info_regression(features, target, n_neighbors=10)
    return np.mean(mi_values)

def compute_mi_classification(features, target):
    """
    Computes the mutual information between a set of continuous features
    (features: shape [N, d]) and a discrete target (class labels, shape [N]).
    Returns the average MI across feature dimensions.
    """
    mi_values = mutual_info_classif(features, target, n_neighbors=10)
    return np.mean(mi_values)

# def compute_entropy(target, bins=30):
#     """
#     Estimate the differential entropy of a 1D continuous target using a histogram.
    
#     Args:
#         target (np.array): 1D array of values.
#         bins (int): number of bins for histogram estimation.
    
#     Returns:
#         entropy (float): estimated differential entropy.
#     """
#     hist, bin_edges = np.histogram(target, bins=bins, density=True)
#     # Only keep nonzero probabilities
#     hist = hist[hist > 0]
#     bin_width = (bin_edges[-1] - bin_edges[0]) / bins
#     entropy = -np.sum(hist * np.log(hist)) * bin_width
#     return entropy

def compute_entropy(target, bins=30):
    hist, bin_edges = np.histogramdd(target, bins=bins, density=True) 
    return entropy(hist.flatten())


# --- Main Experiment ---

def main():
    # Folder containing saved checkpoints.
    checkpoint_folder = "saved_runs/epoch_100"
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pt")))
    
    alphas = []
    mi_class_list = []
    mi_shift_list = []
    entropy_shift_list = []
    fraction_captured_list = []
    # List for class-specific fraction captured (for class 0)
    class_specific_fraction_captured_list = []

    # For normalized MI using negentropy of the representations
    normalized_mi_shift_list = []
    normalized_mi_class_list = []
    entropy_R_list = []
    
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        
        # Each checkpoint is assumed to contain:
        # "alpha", "projected_points", "shifts", and "ids"
        alpha = ckpt["alpha"]
        representations = ckpt["projected_points"]  # shape: [N, d] (e.g. d=2)
        shifts = ckpt["shifts"]                     # shape: [N] or [N, 1]
        ids = ckpt["ids"]                           # shape: [N]

        
        # Convert to numpy arrays.
        representations = representations.clone().detach().float().numpy()
        shifts = shifts.clone().detach().float().numpy().flatten()
        ids = ids.clone().detach().long().numpy()
        
        # Compute MI between representation and class (discrete target)
        mi_class = compute_mi_classification(representations, ids)
        # Compute MI between representation and shift (continuous target)
        mi_shift = compute_mi_regression(representations, shifts)
        
        # Estimate the entropy of the shift distribution.
        H_shift = compute_entropy(shifts, bins=100)
        # Compute the fraction of transformation information captured.
        fraction_captured = mi_shift / H_shift if H_shift > 0 else 0

        selected_indices = np.where(ids == 0)[0]
        selected_representations = representations[selected_indices]
        selected_shifts = shifts[selected_indices]
        # Compute MI for the selected class
        mi_shift_class = compute_mi_regression(selected_representations, selected_shifts)
        H_shift_class = compute_entropy(selected_shifts, bins=100)
        class_specific_fraction_captured = mi_shift_class / H_shift_class if H_shift_class > 0 else 0
    
        # Compute negentropy of the representations.
        H_R = compute_entropy(representations, bins=100)
        # Compute normalized MI using negentropy as the information budget.
        normalized_mi_shift = mi_shift / H_R if H_R > 0 else 0
        normalized_mi_class = mi_class / H_R if H_R > 0 else 0
        
        alphas.append(alpha)
        mi_class_list.append(mi_class)
        mi_shift_list.append(mi_shift)
        entropy_shift_list.append(H_shift)
        fraction_captured_list.append(fraction_captured)
        class_specific_fraction_captured_list.append(class_specific_fraction_captured)
        normalized_mi_shift_list.append(normalized_mi_shift)
        normalized_mi_class_list.append(normalized_mi_class)
        entropy_R_list.append(H_R)
        
        print(f"File: {ckpt_file} | alpha: {alpha:.2f} | MI(Class): {mi_class:.3f} | MI(Shift): {mi_shift:.3f} | H(Shift): {H_shift:.3f} | Fraction: {fraction_captured:.3f} | Class0 Fraction: {class_specific_fraction_captured:.3f} | H(R): {H_R:.3f}")
    
    alphas = np.array(alphas)
    mi_class_array = np.array(mi_class_list)
    mi_shift_array = np.array(mi_shift_list)
    fraction_array = np.array(fraction_captured_list)
    class_specific_fraction_array = np.array(class_specific_fraction_captured_list)
    normalized_mi_shift_array = np.array(normalized_mi_shift_list)
    normalized_mi_class_array = np.array(normalized_mi_class_list)
    entropy_R_array = np.array(entropy_R_list)

    print(class_specific_fraction_array)
    print(mi_shift_array)
    
    # --- Plotting ---
    
    # Plot MI between representation and class vs. alpha.
    fig1, ax1 = plt.subplots()
    ax1.plot(alphas, mi_class_array, marker="o", linestyle="-")
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Mutual Information (Class)")
    ax1.set_title("MI between Representation and Class vs. Alpha")
    
    # Plot MI between representation and shift vs. alpha.
    fig2, ax2 = plt.subplots()
    ax2.plot(alphas, mi_shift_array, marker="o", linestyle="-", color="orange")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Mutual Information (Shift)")
    ax2.set_title("MI between Representation and Shift vs. Alpha")
    
    # Plot the fraction of transformation information captured vs. alpha.
    fig3, ax3 = plt.subplots()
    ax3.plot(alphas, fraction_array, marker="o", linestyle="-", color="green")
    ax3.set_xlabel("Alpha")
    ax3.set_ylabel("Fraction (MI(Shift)/H(Shift))")
    ax3.set_title("Fraction of Transformation Information Captured vs. Alpha")
    
    # Plot the ratio of MI(Class) / MI(Shift) vs. alpha.
    ratio_array = mi_class_array / mi_shift_array
    fig4, ax4 = plt.subplots()
    ax4.plot(alphas, ratio_array, marker="o", linestyle="-", color="red")
    ax4.set_xlabel("Alpha")
    ax4.set_ylabel("MI(Class) / MI(Shift)")
    ax4.set_title("Ratio of MI(Class) to MI(Shift) vs. Alpha")
    
    # Plot the class-specific fraction for class 0 vs. alpha.
    fig5, ax5 = plt.subplots()
    ax5.plot(alphas, class_specific_fraction_array, marker="o", linestyle="-", color="purple")
    ax5.set_xlabel("Alpha")
    ax5.set_ylabel("Fraction (Class0 MI(Shift)/H(Class0 Shift))")
    ax5.set_title("Class 0: Fraction of Transformation Information Captured vs. Alpha")
    
    # --- New Plot: Normalized MI using Negentropy ---
    fig6, ax6 = plt.subplots()
    ax6.plot(alphas, normalized_mi_shift_array, marker="o", linestyle="-", label="I(R;T)/H(R)")
    ax6.plot(alphas, normalized_mi_class_array, marker="o", linestyle="-", label="I(R;C)/H(R)")
    ax6.plot(alphas, entropy_R_array, marker="o", linestyle="-", label="H(R)")
    ax6.set_xlabel("Alpha")
    ax6.set_ylabel("Normalized Mutual Information ")
    ax6.set_title("Normalized MI vs. Alpha)")
    ax6.legend()
    
    # --- Log Figures to TensorBoard ---
    writer = SummaryWriter(log_dir="lightning_logs/contrastive_model_logs/MIAnalysis", comment="MI Analysis with Negentropy")
    writer.add_figure("MI/Class_vs_Alpha", fig1, global_step=0)
    writer.add_figure("MI/Shift_vs_Alpha", fig2, global_step=0)
    writer.add_figure("MI/Fraction_Captured_vs_Alpha", fig3, global_step=0)
    writer.add_figure("MI/Ratio_Class_Shift_vs_Alpha", fig4, global_step=0)
    writer.add_figure("MI/Class0_Fraction_Captured_vs_Alpha", fig5, global_step=0)
    writer.add_figure("MI/Normalized_MI_vs_Alpha", fig6, global_step=0)
    writer.close()
    
    plt.show()

if __name__ == "__main__":
    main()
