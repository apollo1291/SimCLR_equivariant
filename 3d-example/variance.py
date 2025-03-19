import os
import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

# --- Utility Functions ---

def linear_regression_explained_variance(X, Y):
    """
    Given predictors X (shape: [N, p]) and responses Y (shape: [N, d]),
    fit a linear model Y_hat = X beta using torch.linalg.lstsq and compute
    the explained variance (R^2) for each dimension in Y.
    
    Returns the average R^2 across output dimensions.
    """
    # Add an intercept term to X.
    ones = torch.ones(X.shape[0], 1, device=X.device)
    X_ = torch.cat([ones, X], dim=1)  # [N, p+1]

    # Solve the least squares problem: beta = argmin ||X_ beta - Y||_2
    # Using torch.linalg.lstsq which returns a solution in .solution
    lstsq_out = torch.linalg.lstsq(X_, Y)
    beta = lstsq_out.solution  # shape: [p+1, d]

    # Compute predictions
    Y_pred = X_.mm(beta)  # shape: [N, d]

    # Compute R^2 per dimension: 1 - SS_res/SS_tot
    ss_res = ((Y - Y_pred) ** 2).sum(dim=0)
    ss_tot = ((Y - Y.mean(dim=0)) ** 2).sum(dim=0).clamp(min=1e-8)
    r2 = 1 - ss_res / ss_tot

    return r2.mean().item()

def compute_explained_variance_for_shift(representations, shifts):
    """
    representations: Tensor of shape [N, d]
    shifts: Tensor of shape [N] or [N, 1]
    
    Returns the average explained variance (R^2) when regressing the representation on shifts.
    """
    if shifts.dim() == 1:
        shifts = shifts.view(-1, 1)
    return linear_regression_explained_variance(shifts, representations)

def compute_explained_variance_for_class(representations, ids):
    """
    representations: Tensor of shape [N, d]
    ids: LongTensor of shape [N] containing class labels
    
    Converts ids to one-hot encoding and regresses representations on it.
    Returns the average explained variance (R^2).
    """
    num_classes = int(ids.max().item()) + 1
    X_onehot = F.one_hot(ids, num_classes=num_classes).float()
    return linear_regression_explained_variance(X_onehot, representations)

# --- Main Experiment ---

def main():
    # Folder containing saved checkpoints.
    checkpoint_folder = "saved_runs/epoch_100"
    checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_folder, "*.pt")))
    
    alphas = []
    ev_class_list = []
    ev_shift_list = []
    
    # variance of the dataset over all the files 
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        
        # Assume the checkpoint dictionary contains:
        # "alpha", "projected_points", "shifts", and "ids"
        alpha = ckpt["alpha"]
        representations = ckpt["projected_points"]  # shape [N, d]
        shifts = ckpt["shifts"]                     # shape [N] or [N, 1]
        ids = ckpt["ids"]                           # shape [N]

        
        # Convert to torch tensors if necessary.
        if not torch.is_tensor(representations):
            representations = torch.tensor(representations)
        if not torch.is_tensor(shifts):
            shifts = torch.tensor(shifts)
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, dtype=torch.long)
        
        # Compute explained variance (R^2)
        ev_class = compute_explained_variance_for_class(representations, ids)
        ev_shift = compute_explained_variance_for_shift(representations, shifts)
        
        alphas.append(alpha)
        ev_class_list.append(ev_class)
        ev_shift_list.append(ev_shift)
        
        print(f"File: {ckpt_file} | alpha: {alpha:.2f} | EV(class): {ev_class:.3f} | EV(shift): {ev_shift:.3f}")
    
    # --- Plotting ---
    alphas_tensor = torch.tensor(alphas)
    ev_class_tensor = torch.tensor(ev_class_list)
    ev_shift_tensor = torch.tensor(ev_shift_list)
    

    # variance of a single class over all the files
    interested_class = 0
    ev_class_specific_shift_list = []
    for ckpt_file in checkpoint_files:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        alpha = ckpt["alpha"]
        representations = ckpt["projected_points"] 
        shifts = ckpt["shifts"]                     
        ids = ckpt["ids"]   

        selected_points = torch.nonzero(torch.where(ids == interested_class, 1, 0))

        selected_representations = representations[selected_points].squeeze(1)
        selected_shift = shifts[selected_points]

        # single class so the rest is just random noise
        ev_shift = compute_explained_variance_for_shift(selected_representations, selected_shift)
        ev_class_specific_shift_list.append(ev_shift)

    ev_class_specific_shift_tensor = torch.tensor(ev_class_specific_shift_list)


    fig1, ax1 = plt.subplots()
    ax1.plot(alphas_tensor.numpy(), ev_class_tensor.numpy(), marker="o", linestyle="-")
    ax1.set_xlabel("Alpha")
    ax1.set_ylabel("Explained Variance (Class)")
    ax1.set_title("EV of Representation w.r.t. Class")
    
    # Plot explained variance for shift vs. alpha
    fig2, ax2 = plt.subplots()
    ax2.plot(alphas_tensor.numpy(), ev_shift_tensor.numpy(), marker="o", linestyle="-", color="orange")
    ax2.set_xlabel("Alpha")
    ax2.set_ylabel("Explained Variance (Shift)")
    ax2.set_title("EV of Representation w.r.t. Shift")
    
    fig3, ax3 = plt.subplots()
    ax3.plot(alphas_tensor.numpy(), ev_class_specific_shift_tensor.numpy(), marker="o", linestyle="-", color="red")
    ax3.set_xlabel("Alpha")
    ax3.set_ylabel(f"Explained Variance (Shift) of Class {interested_class}")
    ax3.set_title(f"EV of Class {interested_class} Representation w.r.t. Shift")

    writer = SummaryWriter(log_dir="lightning_logs/contrastive_model_logs/VarAnalysis", comment="Variance Analysis")
    writer.add_figure("EV/Class_vs_Alpha", fig1, global_step=0)
    writer.add_figure("EV/Shift_vs_Alpha", fig2, global_step=0)
    writer.add_figure(f"EV/Shift_of_Class_{interested_class}_vs_Alpha", fig3, global_step=0)
    writer.close()


if __name__ == "__main__":
    main()