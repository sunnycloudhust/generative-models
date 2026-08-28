from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


class FlatImageDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), 0


def _build_transform(image_size, is_training):
    """Create the image preprocessing pipeline for one data split."""
    if is_training:
        image_transforms = [
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        image_transforms = [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
        ]

    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(image_transforms)


def _build_dataset(image_root, transform):
    """Use class-aware loading for nested folders and flat loading otherwise."""
    has_class_folders = any(path.is_dir() for path in image_root.iterdir())
    if has_class_folders:
        return datasets.DatasetFolder(
            str(image_root),
            loader=lambda path: Image.open(path).convert("RGB"),
            extensions=(".jpg", ".jpeg", ".png"),
            transform=transform,
        )
    return FlatImageDataset(image_root, transform)


def build_loader(data_root, image_size, batch_size, workers, split="train"):
    """Build a DataLoader for flat or class-organized image folders."""
    image_root = Path(data_root)
    if not image_root.is_dir():
        raise FileNotFoundError(f"Expected image directory at {image_root}")

    is_training = split == "train"
    transform = _build_transform(image_size, is_training)
    dataset = _build_dataset(image_root, transform)
    if not len(dataset):
        raise FileNotFoundError(f"No image files found under {image_root}.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=is_training,
    )
