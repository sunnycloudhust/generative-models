"""Loss functions for the VAE."""

from __future__ import annotations

from torch import Tensor, nn


def vae_loss(
    reconstructions: Tensor, images: Tensor, mu: Tensor, logvar: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    reconstruction_loss = nn.functional.binary_cross_entropy(
        reconstructions, images, reduction="sum"
    )
    kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
    total_loss = reconstruction_loss + kl_loss
    return total_loss, reconstruction_loss, kl_loss