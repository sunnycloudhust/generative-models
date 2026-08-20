import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image

from config import CONFIG
from data import build_loader
from unet import UNetVelocity


@torch.no_grad()
def sample(model, device, n_samples, image_size, steps):
    """
    This function samples (n_samples) images from noise distribution using ODE
    """
    model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n_samples,), i * dt, device=device)
        x = x + model(x, t) * dt
    return x.clamp(-1, 1)


def save_samples(model, device, out_dir, step, image_size, n_samples, sample_steps):
    """
        This function samples the image from noise distribution every sample_steps
    """
    x = sample(model, device, n_samples, image_size, sample_steps)
    grid = make_grid((x + 1) * 0.5, nrow=int(math.sqrt(n_samples)))
    save_image(grid, out_dir / f"samples_step_{step:07d}.png")


def train(config):
    device_name = config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(
        data_root=config["data_root"],
        batch_size=config["batch_size"],
        image_size=config["image_size"],
        workers=config["workers"],
    )
    model = UNetVelocity(base_channels=config["base_channels"]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    use_amp = config["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    print(f"device={device}")
    print(f"dataset_size={len(loader.dataset):,}")
    print(f"parameters={sum(p.numel() for p in model.parameters()):,}")

    step = 0
    while step < config["steps"]:
        for x1, _ in loader:
            step += 1
            x1 = x1.to(device, non_blocking=True)
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            t_img = t[:, None, None, None]

            x_t = (1 - t_img) * x0 + t_img * x1
            target_v = x1 - x0

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_v = model(x_t, t)
                loss = F.mse_loss(pred_v, target_v)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            scaler.step(opt)
            scaler.update()

            if step % config["log_every"] == 0:
                print(f"step={step:07d}/{config['steps']} loss={loss.item():.5f}")

            if step % config["sample_every"] == 0:
                save_samples(
                    model,
                    device,
                    out_dir,
                    step,
                    config["image_size"],
                    config["sample_n"],
                    config["sample_steps"],
                )
                model.train()

            if step % config["ckpt_every"] == 0:
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "config": config,
                    },
                    out_dir / f"ckpt_step_{step:07d}.pt",
                )

            if step >= config["steps"]:
                break

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    save_samples(
        model,
        device,
        out_dir,
        step,
        config["image_size"],
        config["sample_n"],
        config["sample_steps"],
    )


if __name__ == "__main__":
    train(CONFIG)
