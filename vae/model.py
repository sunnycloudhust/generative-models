"""VAE model definition."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VAE(nn.Module):
    def __init__(self, latent_dim: int = 20) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(28 * 28, 400), nn.ReLU())
        self.mu = nn.Linear(400, latent_dim)
        self.logvar = nn.Linear(400, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 28 * 28),
            nn.Sigmoid(),
        )

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encoder(images.view(images.size(0), -1))
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        standard_deviation = torch.exp(0.5 * logvar)
        noise = torch.randn_like(standard_deviation)
        return mu + noise * standard_deviation

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent).view(-1, 1, 28, 28)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(images)
        latent = self.reparameterize(mu, logvar)
        return self.decode(latent), mu, logvar