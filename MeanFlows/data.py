from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
    """Create the image preprocessing pipeline"""
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


def build_loader(data_root, image_size, batch_size, workers, split="train"):
    """Build a DataLoader for images stored directly under data_root."""
    image_root = Path(data_root)
    
    if not image_root.is_dir():
        raise FileNotFoundError(f"Expected image directory at {image_root}")

    is_training = split == "train"
    transform = _build_transform(image_size, is_training)
    dataset = FlatImageDataset(image_root, transform)
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
