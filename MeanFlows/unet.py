import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, values):
        half = self.dim // 2
        if half == 0:
            return values[:, None]
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=values.device, dtype=values.dtype)
            / max(half - 1, 1)
        )
        angles = values[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(min(32, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_embedding):
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_proj(time_embedding)[:, :, None, None]
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + self.skip(x)


class MeanFlowUNet(nn.Module):
    """U-Net that predicts the average velocity u(x_r, r, t)."""

    def __init__(self, in_channels=3, base_channels=128):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embedding = SinusoidalEmbedding(base_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels * 2, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c1, c2, c3, c4 = (
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 4,
        )
        self.input = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(c1, c1, time_dim), ResBlock(c1, c1, time_dim)])
        self.down1 = nn.Conv2d(c1, c1, 4, stride=2, padding=1)
        self.enc2 = nn.ModuleList([ResBlock(c1, c2, time_dim), ResBlock(c2, c2, time_dim)])
        self.down2 = nn.Conv2d(c2, c2, 4, stride=2, padding=1)
        self.enc3 = nn.ModuleList([ResBlock(c2, c3, time_dim), ResBlock(c3, c3, time_dim)])
        self.down3 = nn.Conv2d(c3, c3, 4, stride=2, padding=1)
        self.mid = nn.ModuleList([ResBlock(c3, c4, time_dim), ResBlock(c4, c4, time_dim)])
        self.up3 = nn.Conv2d(c4, c4, 3, padding=1)
        self.dec3 = nn.ModuleList([ResBlock(c4 + c3, c3, time_dim), ResBlock(c3, c3, time_dim)])
        self.up2 = nn.Conv2d(c3, c3, 3, padding=1)
        self.dec2 = nn.ModuleList([ResBlock(c3 + c2, c2, time_dim), ResBlock(c2, c2, time_dim)])
        self.up1 = nn.Conv2d(c2, c2, 3, padding=1)
        self.dec1 = nn.ModuleList([ResBlock(c2 + c1, c1, time_dim), ResBlock(c1, c1, time_dim)])
        self.output = nn.Sequential(
            nn.GroupNorm(min(32, c1), c1), nn.SiLU(), nn.Conv2d(c1, in_channels, 3, padding=1)
        )

    @staticmethod
    def _blocks(blocks, x, time_embedding):
        for block in blocks:
            x = block(x, time_embedding)
        return x

    def forward(self, x, r, t):
        time_embedding = self.time_mlp(
            torch.cat((self.time_embedding(r), self.time_embedding(t)), dim=-1)
        )
        hidden = self.input(x)
        skip1 = self._blocks(self.enc1, hidden, time_embedding)
        hidden = self.down1(skip1)
        skip2 = self._blocks(self.enc2, hidden, time_embedding)
        hidden = self.down2(skip2)
        skip3 = self._blocks(self.enc3, hidden, time_embedding)
        hidden = self.down3(skip3)
        hidden = self._blocks(self.mid, hidden, time_embedding)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.up3(hidden)
        hidden = self._blocks(self.dec3, torch.cat((hidden, skip3), dim=1), time_embedding)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.up2(hidden)
        hidden = self._blocks(self.dec2, torch.cat((hidden, skip2), dim=1), time_embedding)
        hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.up1(hidden)
        hidden = self._blocks(self.dec1, torch.cat((hidden, skip1), dim=1), time_embedding)
        return self.output(hidden)
