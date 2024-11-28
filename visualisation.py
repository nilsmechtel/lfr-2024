import torchvision.utils as vutils
import seaborn as sns
import matplotlib.pyplot as plt
import math
import torch


def scale(data, eps=1e-6):
    """
    Perform min-max normalization to scale data to [0, 1] along the last dimension.
    Args:
        data (torch.Tensor): The distilled data.
        eps (float): Small constant to prevent division by zero.
    Returns:
        torch.Tensor: Scaled data.
    """
    # Calculate min and max for each sample in the batch
    sample_mins = data.min(dim=-1, keepdim=True)[0]
    sample_maxs = data.max(dim=-1, keepdim=True)[0]
    return (data - sample_mins) / (sample_maxs - sample_mins + eps)

def create_image_grid(tensor, images_per_row=10, total_rows=8):
    """
    Create an image grid with the given flattened tensor.

    Args:
        tensor (torch.Tensor): Tensor containing flattened images to be arranged in a grid.
                            It should be of shape (batch_size, N*N).
        images_per_row (int, optional): Number of images per row (default is 10).
        total_rows (int, optional): Total number of rows in the grid (default is 8).

    Returns:
        wandb.Image: A grid of images arranged as specified.
    """
    # Disable gradient computation
    with torch.no_grad():
        # Ensure we are working with at most total_rows * images_per_row images
        batch_size = tensor.size(0)
        images_to_show = min(batch_size, total_rows * images_per_row)
        tensor = tensor[:images_to_show]
        
        # Normalize to [0, 1] for visualization
        tensor = scale(tensor)

        # Reshape the flattened images back to their 2D form (height, width)
        original_image_size = int(math.sqrt(tensor.size(-1)))
        tensor = tensor.view(
            -1, 1, original_image_size, original_image_size
        )  # Assuming 1 channel (grayscale)

        # Create the image grid with the specified number of images per row
        image_grid = vutils.make_grid(
            tensor, nrow=images_per_row, normalize=True, padding=2, pad_value=1
        )

    return image_grid


def plot_contingency_table(contingency_table):
    # Plot contingency table as a heatmap with labels
    fig, ax = plt.subplots()
    sns.heatmap(contingency_table, cmap="viridis", ax=ax)
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    contingency_table = torch.rand(10, 10)
    fig = plot_contingency_table(contingency_table)
    plt.savefig("test_contingency_table.png")