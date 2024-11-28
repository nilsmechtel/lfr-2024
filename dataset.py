import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torchvision import datasets, transforms


def prepare_mnist_data(batch_size=128, val_split=0.1, num_workers=10):
    """
    Prepare MNIST dataset for training and validation.
    - Images are normalized to [0, 1] and flattened to 784-dimensional vectors.
    - Labels are one-hot encoded using torch.nn.functional.one_hot.

    Args:
        batch_size (int): Batch size for data loaders.
        val_split (float): Fraction of data reserved for validation.

    Returns:
        train_loader (DataLoader): DataLoader for training dataset.
        val_loader (DataLoader): DataLoader for validation dataset.
    """
    # Define transforms for images
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and shape [1, 28, 28]
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to shape [784]
    ])

    # Use target_transform for one-hot encoding labels
    def target_transform(label):
        return F.one_hot(torch.tensor(label), num_classes=10).float()

    # Load MNIST dataset
    mnist_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform, target_transform=target_transform
    )

    # Split into training and validation datasets
    val_size = int(len(mnist_dataset) * val_split)
    train_size = len(mnist_dataset) - val_size
    train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = prepare_mnist_data()

    # Test the data
    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
