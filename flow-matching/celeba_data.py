from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class FlatImageDataset(Dataset):
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.paths = sorted(
            path
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp")
            for path in self.root.glob(pattern)
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image), 0


def build_loader(data_root, batch_size, image_size, workers, download):
    transform = transforms.Compose(
        [
            transforms.CenterCrop(178),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    try:
        dataset = datasets.CelebA(
            root=data_root,
            split="train",
            target_type="attr",
            transform=transform,
            download=download,
        )
    except RuntimeError as err:
        fallback = Path(data_root) / "celeba" / "img_align_celeba"
        if not fallback.exists():
            raise RuntimeError(
                "CelebA download failed or dataset is missing. Put images under "
                f"{fallback}/*.jpg, or set download=True in celeba_config.py."
            ) from err
        dataset = FlatImageDataset(fallback, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
