
import os
import torch
import matplotlib.pyplot as plt
from cluster_example import ContrastiveModel  # adjust import as needed


def validate_saved_runs(checkpoints_folder="saved_runs"):
    """
    Loads each .pt file from `checkpoints_folder`,
    reconstructs the model, and demonstrates usage of
    the saved model and representations.
    """
    # List all saved checkpoints in the folder
    for file_name in sorted(os.listdir(checkpoints_folder)):
        if file_name.endswith(".pt"):
            checkpoint_path = os.path.join(checkpoints_folder, file_name)
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            # Extract the saved items
            alpha = checkpoint["alpha"]
            model_state_dict = checkpoint["model_state_dict"]
            original_points = checkpoint["original_points"]
            projected_points = checkpoint["projected_points"]
            shift = checkpoint["shifts"]
            ids = checkpoint["ids"]
            X = checkpoint["X"]
            X3D = checkpoint["X3D"]

            
            model = ContrastiveModel(
                input_dim=3,
                output_dim=2,
                lr=2e-4,
                alpha=alpha
            )
            model.load_state_dict(model_state_dict)
            model.eval()  # set model to eval mode

            print(f"Loaded checkpoint from: {checkpoint_path}")
            print(f"  - alpha = {alpha:.2f}")
            print(f"  - original_points shape = {original_points.shape}")
            print(f"  - projected_points shape = {projected_points.shape}")
            print(f"  - X (full dataset) shape = {X.shape}")
            print(f"  - X3D (full dataset) shape = {X3D.shape}")
            print(f"  - ids shape = {ids.shape}")

            print(f"  - X vs ids first 10 = {X[:10]}, {ids[:10]}")

            # OPTIONAL: Plot original vs. projected
            # (just to visually confirm that we've recovered the data)
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            # Left plot: original 2D points color-coded by Y
            colors = original_points[:, 1].numpy()
            ax[0].scatter(
                original_points[:, 0],
                original_points[:, 1],
                c=colors,
                alpha=0.5,
                cmap="viridis"
            )
            ax[0].set_title("Original 2D points")

            # Right plot: learned 2D embeddings, color-coded by same Y
            ax[1].scatter(
                projected_points[:, 0],
                projected_points[:, 1],
                c=colors,
                alpha=0.5,
                cmap="viridis"
            )
            ax[1].set_title("Projected 2D embeddings")

            plt.suptitle(f"Validation of Checkpoint with alpha={alpha:.2f}")
            plt.savefig("validation.png")


            with torch.no_grad():
                z1, z2 = model(X3D, shift)
                print(z1.shape, z2.shape)

if __name__ == "__main__":
    validate_saved_runs()