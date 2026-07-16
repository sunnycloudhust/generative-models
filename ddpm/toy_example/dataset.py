from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMG_SIZE = 64
BATCH_SIZE = 128
DATASET_DIR = Path("1_Liner TF")


class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.image_paths = sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in image_exts
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
    data_transform = transforms.Compose(data_transforms)
    return FlatImageDataset(DATASET_DIR, transform=data_transform)


def show_tensor_image(image):
    reverse_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda t: (t + 1) / 2),
            transforms.Lambda(lambda t: t.permute(1, 2, 0)),
            transforms.Lambda(lambda t: t * 255.0),
            transforms.Lambda(lambda t: t.detach().cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )

    if len(image.shape) == 4:
        image = image[0, :, :, :]

    plt.imshow(reverse_transforms(image))
    plt.axis("off")


data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
