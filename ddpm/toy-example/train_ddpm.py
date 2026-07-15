import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm


def exists(x):
    return x is not None


def extract(values, timesteps, x_shape):
    batch_size = timesteps.shape[0]
    out = values.gather(-1, timesteps.cpu()).to(timesteps.device)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def sinusoidal_embedding(timesteps, dim):
    half_dim = dim // 2
    scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -scale)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = h + self.time_mlp(time_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x).reshape(b, c, h * w).transpose(1, 2)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x + residual


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


class UNet(nn.Module):
    def __init__(
        self,
        image_channels=1,
        base_channels=64,
        channel_mults=(1, 2, 4),
        time_dim=256,
        num_heads=4,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.init_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_channels]
        downs = []
        in_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            downs.append(nn.ModuleList([
                ResBlock(in_channels, out_channels, time_dim),
                ResBlock(out_channels, out_channels, time_dim),
                AttentionBlock(out_channels, num_heads),
                Downsample(out_channels) if i != len(channel_mults) - 1 else nn.Identity(),
            ]))
            channels.append(out_channels)
            in_channels = out_channels
        self.downs = nn.ModuleList(downs)

        mid_channels = channels[-1]
        self.mid1 = ResBlock(mid_channels, mid_channels, time_dim)
        self.mid_attn = AttentionBlock(mid_channels, num_heads)
        self.mid2 = ResBlock(mid_channels, mid_channels, time_dim)

        ups = []
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * mult
            skip_channels = channels.pop()
            ups.append(nn.ModuleList([
                ResBlock(in_channels + skip_channels, out_channels, time_dim),
                ResBlock(out_channels, out_channels, time_dim),
                AttentionBlock(out_channels, num_heads),
                Upsample(out_channels) if i != 0 else nn.Identity(),
            ]))
            in_channels = out_channels
        self.ups = nn.ModuleList(ups)

        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, image_channels, 3, padding=1),
        )

    def forward(self, x, timesteps):
        time_emb = sinusoidal_embedding(timesteps, self.time_dim)
        time_emb = self.time_mlp(time_emb)

        x = self.init_conv(x)
        skips = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)

        return self.final(x)


class DDPM:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device

        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = F.pad(alpha_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.alpha_cumprod_prev = alpha_cumprod_prev
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)

    def to(self, device):
        self.device = device
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        return self

    def q_sample(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alpha_cumprod, timesteps, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x_start.shape) * noise
        )

    @torch.no_grad()
    def p_sample(self, model, x, t):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alpha_cumprod_t = extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alpha_cumprod_t
        )
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)

        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=1):
        model.eval()
        x = torch.randn(batch_size, channels, image_size, image_size, device=self.device)
        for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps, desc="Sampling"):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x


def build_dataloader(args):
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset_name = args.dataset.lower()
    if dataset_name == "mnist":
        dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
        channels = 1
    elif dataset_name == "fashionmnist":
        dataset = datasets.FashionMNIST(args.data_dir, train=True, download=True, transform=transform)
        channels = 1
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
        channels = 3
    else:
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
        sample, _ = dataset[0]
        channels = sample.shape[0]

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader, channels


def save_samples(ddpm, model, args, epoch, channels):
    samples = ddpm.sample(
        model,
        image_size=args.image_size,
        batch_size=args.num_samples,
        channels=channels,
    )
    samples = (samples.clamp(-1, 1) + 1) * 0.5
    out_path = Path(args.output_dir) / "samples" / f"epoch_{epoch:04d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(samples, out_path, nrow=int(math.sqrt(args.num_samples)))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    loader, channels = build_dataloader(args)
    model = UNet(
        image_channels=channels,
        base_channels=args.base_channels,
        channel_mults=tuple(args.channel_mults),
        time_dim=args.time_dim,
        num_heads=args.num_heads,
    ).to(device)
    ddpm = DDPM(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    if exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        total_loss = 0.0

        for x, _ in progress:
            x = x.to(device)
            batch_size = x.shape[0]
            t = torch.randint(0, args.timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(x)
            noisy_x = ddpm.q_sample(x, t, noise)
            noise_pred = model(noisy_x, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"epoch={epoch} loss={avg_loss:.6f}")

        if epoch % args.sample_every == 0:
            save_samples(ddpm, model, args, epoch, channels)

        if epoch % args.save_every == 0:
            ckpt_path = Path(args.output_dir) / f"ddpm_epoch_{epoch:04d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                ckpt_path,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM 2020 noise-prediction model.")
    parser.add_argument("--dataset", default="mnist", choices=["mnist", "fashionmnist", "cifar10", "imagefolder"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./runs/ddpm")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--time-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
