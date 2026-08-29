import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import jvp

from config import CONFIG
from data import build_loader
from unet import MeanFlowUNet
from sample import sample
from torchvision.utils import make_grid, save_image


def meanflow_loss(model, x0, x1):
    batch_size = x1.shape[0]
    t = torch.rand(batch_size, device=x1.device)
    r = torch.rand(batch_size, device=x1.device) * t #0<r<t<1
    x_t = x0 + t[:, None, None, None] * (x1 - x0)
    velocity = x1 - x0
    ones = torch.ones_like(t)

    with torch.no_grad():
        _, derivative = jvp(
            lambda state, time: model(state, r, time),
            (x_t, t),
            (velocity, ones),
        )
    prediction = model(x_t, r, t)
    target = (velocity - (t - r)[:, None, None, None] * derivative).detach()
    return torch.mean((prediction - target) ** 2)


def save_samples(model, device, output_dir, step, config):
    images = sample(model, device, config["sample_n"], config["image_size"], config["sample_steps"])
    grid = make_grid((images + 1.0) * 0.5, nrow=max(1, int(config["sample_n"] ** 0.5)))
    save_image(grid, output_dir / f"samples_step_{step}.png")


def build_scaler(device, enabled):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=True)
    if device.type == "cuda":
        return torch.cuda.amp.GradScaler(enabled=True)
    return None


def train(config, resume=None):
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(config["out_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = build_loader(config["data_root"], config["image_size"], config["batch_size"], config["workers"])
    model = MeanFlowUNet(base_channels=config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    use_amp = bool(config["amp"] and device.type == "cuda")
    scaler = build_scaler(device, use_amp)
    
    step = 0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        step = state.get("step", 0)
    print(f"device={device} images={len(loader.dataset):,} parameters={sum(p.numel() for p in model.parameters()):,}")
    model.train()
    while step < config["steps"]:
        for images, _ in loader:
            step += 1
            images = images.to(device, non_blocking=True)
            noise = torch.randn_like(images)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = meanflow_loss(model, noise, images)
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
            
            if step % config["log_every"] == 0:
                print(f"step={step}/{config['steps']} loss={loss.item():.5f}")
            if step % config["sample_every"] == 0:
                save_samples(model, device, output_dir, step, config)
            if step % config["ckpt_every"] == 0:
                checkpoint = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config}
                if scaler is not None:
                    checkpoint["scaler"] = scaler.state_dict()
                torch.save(checkpoint, output_dir / f"ckpt_step_{step}.pt")
            if step >= config["steps"]:
                break
    final_checkpoint = {"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config}
    if scaler is not None:
        final_checkpoint["scaler"] = scaler.state_dict()
    torch.save(final_checkpoint, output_dir / "model_final.pt")
    save_samples(model, device, output_dir, step, config)


if __name__ == "__main__":
    train(CONFIG, CONFIG["resume"])
