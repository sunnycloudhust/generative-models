import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class FlowMatchingModelConfig:
    def __init__(self, in_channels=3, base_channels=64):
        self.in_channels = in_channels
        self.base_channels = base_channels

    def to_dict(self):
        return {
            "model_type": "flow_matching_unet",
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "architecture": "UNetVelocity",
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            in_channels=data.get("in_channels", 3),
            base_channels=data.get("base_channels", 64),
        )


class FlowMatchingModel(nn.Module):
    def __init__(self, config=None, in_channels=3, base_channels=64):
        super().__init__()
        if config is None:
            config = FlowMatchingModelConfig(
                in_channels=in_channels,
                base_channels=base_channels,
            )
        self.config = config

        time_dim = self.config.base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(self.config.base_channels),
            nn.Linear(self.config.base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c1 = self.config.base_channels
        c2 = self.config.base_channels * 2
        c3 = self.config.base_channels * 4
        c4 = self.config.base_channels * 4

        self.init_conv = nn.Conv2d(self.config.in_channels, c1, 3, padding=1)
        self.enc1 = nn.ModuleList(
            [ResBlock(c1, c1, time_dim), ResBlock(c1, c1, time_dim)]
        )
        self.down1 = Downsample(c1)
        self.enc2 = nn.ModuleList(
            [ResBlock(c1, c2, time_dim), ResBlock(c2, c2, time_dim)]
        )
        self.down2 = Downsample(c2)
        self.enc3 = nn.ModuleList(
            [ResBlock(c2, c3, time_dim), ResBlock(c3, c3, time_dim)]
        )
        self.down3 = Downsample(c3)

        self.mid1 = ResBlock(c3, c4, time_dim)
        self.mid2 = ResBlock(c4, c4, time_dim)

        self.up3 = Upsample(c4)
        self.dec3 = nn.ModuleList(
            [ResBlock(c4 + c3, c3, time_dim), ResBlock(c3, c3, time_dim)]
        )
        self.up2 = Upsample(c3)
        self.dec2 = nn.ModuleList(
            [ResBlock(c3 + c2, c2, time_dim), ResBlock(c2, c2, time_dim)]
        )
        self.up1 = Upsample(c2)
        self.dec1 = nn.ModuleList(
            [ResBlock(c2 + c1, c1, time_dim), ResBlock(c1, c1, time_dim)]
        )

        self.out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, self.config.in_channels, 3, padding=1),
        )

    def run_blocks(self, blocks, x, temb):
        for block in blocks:
            x = block(x, temb)
        return x

    def forward(self, x, t):
        temb = self.time_mlp(t)
        h = self.init_conv(x)

        s1 = self.run_blocks(self.enc1, h, temb)
        h = self.down1(s1)
        s2 = self.run_blocks(self.enc2, h, temb)
        h = self.down2(s2)
        s3 = self.run_blocks(self.enc3, h, temb)
        h = self.down3(s3)

        h = self.mid2(self.mid1(h, temb), temb)

        h = self.up3(h)
        h = self.run_blocks(self.dec3, torch.cat([h, s3], dim=1), temb)
        h = self.up2(h)
        h = self.run_blocks(self.dec2, torch.cat([h, s2], dim=1), temb)
        h = self.up1(h)
        h = self.run_blocks(self.dec1, torch.cat([h, s1], dim=1), temb)

        return self.out(h)

    @classmethod
    def from_pretrained(cls, model_path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model path does not exist: {path}")

        config_path = path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config.json in {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = FlowMatchingModelConfig.from_dict(data)
        model = cls(config=config)

        weight_file = path / "pytorch_model.bin"
        if not weight_file.exists():
            weight_file = path / "model_final.pt"
        if not weight_file.exists():
            raise FileNotFoundError(f"No weights found in {path}")

        state = torch.load(weight_file, map_location="cpu")
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        model.load_state_dict(state)
        model.eval()
        return model

    def save_pretrained(self, save_directory):
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        torch.save(self.state_dict(), path / "pytorch_model.bin")
        return str(path)


UNetVelocity = FlowMatchingModel
