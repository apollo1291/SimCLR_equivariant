import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility.
np.random.seed(0)
torch.manual_seed(0)

def generate_synthetic_data(N=1000, num_clusters=3, dim=4):
    """
    Generate N 4D points from a mixture of num_clusters Gaussians.
    The first 3 dimensions carry the cluster structure; the 4th is initially zero.
    
    Returns:
        X: np.array, shape [N, 4] of original points.
        ids: np.array, shape [N] of cluster labels.
    """
    points = []
    labels = []
    N_per_cluster = N // num_clusters
    # Define centers for the clusters in the first 3 dimensions; set 4th coordinate to 0.
    centers = np.array([
        [0, 0, 0],
        [5, 5, 0],
        [-5, 5, 0]
    ])
    cov = np.eye(3)  # moderate covariance in first 3 dims
    for i in range(num_clusters):
        pts = np.random.multivariate_normal(mean=centers[i], cov=cov, size=N_per_cluster)
        # Append a zero 4th dimension.
        pts_4d = np.hstack((pts, np.zeros((N_per_cluster, 1))))
        points.append(pts_4d)
        labels.append(np.full(N_per_cluster, i))
    X = np.vstack(points)
    ids = np.concatenate(labels)
    # If needed, shuffle the dataset.
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], ids[idx]

def apply_transformation(X, alpha):
    """
    For each sample x in X (shape [N, 4]), sample a scalar t uniformly from [-1,1]
    and compute the representation:
    
      f(x, t; alpha) = x + (1 - alpha) * t * e4,
      
    where e4 = [0, 0, 0, 1].
    
    Returns:
        projected_points: np.array, shape [N, 4]
        shifts: np.array, shape [N]
    """
    N = X.shape[0]
    # Sample transformation parameters t ~ Uniform(-1, 1)
    shifts = np.random.uniform(low=-1.0, high=1.0, size=N)
    # e4 is the unit vector along the 4th dimension.
    e4 = np.array([0, 0, 0, 1])
    # Compute f(x, t; alpha) = x + (1 - alpha) * t * e4.
    # We use broadcasting: shifts[:, None] multiplies e4.
    projected_points = X + (1 - alpha) * shifts[:, None] * e4
    return projected_points, shifts

def save_checkpoint(alpha, X, projected_points, shifts, ids, save_dir="saved_runs"):
    """
    Save a checkpoint dictionary with keys:
      "alpha", "X", "projected_points", "shifts", "ids"
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    checkpoint = {
        "alpha": alpha,
        "X": torch.tensor(X, dtype=torch.float32),
        "projected_points": torch.tensor(projected_points, dtype=torch.float32),
        "shifts": torch.tensor(shifts, dtype=torch.float32),
        "ids": torch.tensor(ids, dtype=torch.long)
    }
    filename = os.path.join(save_dir, f"model_alpha_{alpha:.2f}.pt")
    torch.save(checkpoint, filename)
    print(f"Saved checkpoint for alpha={alpha:.2f} to {filename}")

def main():
    # Generate synthetic 4D data with cluster structure.
    X, ids = generate_synthetic_data(N=1000, num_clusters=3, dim=4)
    
    # For visualization, let's plot the first 3 dimensions (or a projection) of the original points.
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=ids, cmap="viridis", alpha=0.6)
    plt.title("Original Points (First 2 dimensions)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.savefig("visual.png")
    
    os.makedirs("saved_runs/four_d", exist_ok=True)
    # Now, create and save checkpoints for a range of alpha values.
    alphas = np.linspace(0, 1, 6)  # e.g., 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
    
    for alpha in alphas:
        projected_points, shifts = apply_transformation(X, alpha)
        save_checkpoint(alpha, X, projected_points, shifts, ids, save_dir="saved_runs/four_d")
    
if __name__ == "__main__":
    main()
