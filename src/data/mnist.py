import torch
from torchvision import datasets, transforms

def load_mnist(classes: list[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], batch_size: int = 64, n_images: int = 5000, discretize: bool = True):
    """
    Load the MNIST dataset.

    Parameters:
    classes: [list] 
        List of classes to load.
    batch_size: [int] 
        Batch size of the dataloader.
    n_images: [int]
        Number of images to load per class.
    """
    # Load the MNIST dataset
    dataset = datasets.MNIST(
        root="data/",
        train=True,
        download=True
    )

    # Filter the dataset to only include the specified classes
    mask = torch.zeros(len(dataset.targets), dtype=torch.bool)
    for c in classes:
        mask += (dataset.targets == c)
    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]

    # Take a subset of the dataset
    dataset.data = dataset.data[:n_images]
    dataset.targets = dataset.targets[:n_images]

    # Normalize the dataset
    dataset.data = (dataset.data.float() / 255.0).unsqueeze(1)

    # Discretize the dataset
    if discretize:
        dataset.data = (dataset.data > 0.5).float()

    data_subset = torch.utils.data.TensorDataset(dataset.data, dataset.targets)

    # Create a dataloader
    dataloader = torch.utils.data.DataLoader(
        data_subset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader
