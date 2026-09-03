"""Training loop for one VAE epoch."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from losses import vae_loss
from model import VAE


def train_epoch(
    model: VAE, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device
) -> tuple[float, float, float]:
    model.train()
    total_loss = total_reconstruction = total_kl = 0.0

    for images, _ in loader:
        images = images.to(device)
        optimizer.zero_grad()
        reconstructions, mu, logvar = model(images)
        loss, reconstruction_loss, kl_loss = vae_loss(
            reconstructions, images, mu, logvar
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_reconstruction += reconstruction_loss.item()
        total_kl += kl_loss.item()

    sample_count = len(loader.dataset)
    return (
        total_loss / sample_count,
        total_reconstruction / sample_count,
        total_kl / sample_count,
    )