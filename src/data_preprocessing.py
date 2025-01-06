import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_data_loaders(batch_size=64, download=True):
    """Download and prepare the MNIST dataset."""

    # Define transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
        ]
    )

    # Download and load the training data
    train_dataset = datasets.MNIST(
        root="data/raw", train=True, download=download, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    test_dataset = datasets.MNIST(
        root="data/raw", train=False, download=download, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    # Example usage
    train_loader, test_loader = get_data_loaders(batch_size=64)
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
