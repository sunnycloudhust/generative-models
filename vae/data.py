"""MNIST data loading."""

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def create_mnist_loader(
    data_dir: Path, batch_size: int, shuffle: bool = True
) -> DataLoader:
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)