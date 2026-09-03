"""Save VAE reconstructions and random samples."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from model import VAE


@torch.no_grad()
def save_reconstructions(
    model: VAE, loader: DataLoader, device: torch.device, output_dir: Path, epoch: int
) -> None:
    model.eval()
    images, _ = next(iter(loader))
    images = images[:16].to(device)
    reconstructions, _, _ = model(images)
    comparison = torch.cat((images.cpu(), reconstructions.cpu()))
    save_image(comparison, output_dir / f"reconstruction_epoch_{epoch:03d}.png", nrow=16)


@torch.no_grad()
def save_samples(model: VAE, device: torch.device, output_dir: Path, epoch: int) -> None:
    model.eval()
    latent = torch.randn(64, model.latent_dim, device=device)
    save_image(model.decode(latent).cpu(), output_dir / f"samples_epoch_{epoch:03d}.png", nrow=8)