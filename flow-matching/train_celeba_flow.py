import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
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


class UNetVelocity(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 4

        self.init_conv = nn.Conv2d(in_channels, c1, 3, padding=1)
        self.enc1 = nn.ModuleList([ResBlock(c1, c1, time_dim), ResBlock(c1, c1, time_dim)])
        self.down1 = Downsample(c1)
        self.enc2 = nn.ModuleList([ResBlock(c1, c2, time_dim), ResBlock(c2, c2, time_dim)])
        self.down2 = Downsample(c2)
        self.enc3 = nn.ModuleList([ResBlock(c2, c3, time_dim), ResBlock(c3, c3, time_dim)])
        self.down3 = Downsample(c3)

        self.mid1 = ResBlock(c3, c4, time_dim)
        self.mid2 = ResBlock(c4, c4, time_dim)

        self.up3 = Upsample(c4)
        self.dec3 = nn.ModuleList([ResBlock(c4 + c3, c3, time_dim), ResBlock(c3, c3, time_dim)])
        self.up2 = Upsample(c3)
        self.dec2 = nn.ModuleList([ResBlock(c3 + c2, c2, time_dim), ResBlock(c2, c2, time_dim)])
        self.up1 = Upsample(c2)
        self.dec1 = nn.ModuleList([ResBlock(c2 + c1, c1, time_dim), ResBlock(c1, c1, time_dim)])

        self.out = nn.Sequential(
            nn.GroupNorm(8, c1),
            nn.SiLU(),
            nn.Conv2d(c1, in_channels, 3, padding=1),
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
                f"{fallback}/*.jpg, or run with --download if torchvision can access "
                "the dataset."
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


@torch.no_grad()
def sample(model, device, n_samples, image_size, steps):
    model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples,), i * dt, device=device)
        x = x + model(x, t) * dt
    return x.clamp(-1, 1)


def save_samples(model, device, out_dir, step, image_size, n_samples=16, sample_steps=100):
    x = sample(model, device, n_samples, image_size, sample_steps)
    grid = make_grid((x + 1) * 0.5, nrow=int(math.sqrt(n_samples)))
    save_image(grid, out_dir / f"samples_step_{step:07d}.png")


def train(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args.data_root, args.batch_size, args.image_size, args.workers, args.download)
    model = UNetVelocity(base_channels=args.base_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    print(f"device={device}")
    print(f"dataset_size={len(loader.dataset):,}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")

    step = 0
    while step < args.steps:
        for x1, _ in loader:
            step += 1
            x1 = x1.to(device, non_blocking=True)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            t_img = t[:, None, None, None]

            x_t = (1 - t_img) * x0 + t_img * x1
            target_v = x1 - x0

            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                pred_v = model(x_t, t)
                loss = F.mse_loss(pred_v, target_v)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % args.log_every == 0:
                print(f"step={step:07d}/{args.steps} loss={loss.item():.5f}")

            if step % args.sample_every == 0:
                save_samples(model, device, out_dir, step, args.image_size, args.sample_n, args.sample_steps)
                model.train()

            if step % args.ckpt_every == 0:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "args": vars(args),
                    },
                    out_dir / f"ckpt_step_{step:07d}.pt",
                )

            if step >= args.steps:
                break

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    save_samples(model, device, out_dir, step, args.image_size, args.sample_n, args.sample_steps)


def parse_args():
    parser = argparse.ArgumentParser(description="Train conditional flow matching on CelebA.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--out-dir", default="./runs/celeba_flow")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--sample-n", type=int, default=16)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
