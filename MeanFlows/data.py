from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.5, 0.5, 0.5)
IMAGENET_STD = (0.5, 0.5, 0.5)


def build_loader(data_root, image_size, batch_size, workers, split="train"):
    root = Path(data_root) / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"Expected ImageNet split at {root}. Set --data-root to the directory containing train/ and val/."
        )
    transform_list = [
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0))
        if split == "train"
        else transforms.Resize(image_size + 32),
    ]
    if split != "train":
        transform_list.append(transforms.CenterCrop(image_size))
    transform_list.extend(
        [transforms.RandomHorizontalFlip() if split == "train" else transforms.Lambda(lambda image: image),
         transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    dataset = datasets.ImageFolder(root, transform=transforms.Compose(transform_list))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=split == "train",
    )
