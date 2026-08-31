import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.func import jvp

from config import CONFIG
from data import build_loader
from unet import MeanFlowUNet
from sample import sample
from plot import plot_loss_curve
from torchvision.utils import make_grid, save_image


def meanflow_loss(model, x0, x1):
    x0 = x0.float()
    x1 = x1.float()
    batch_size = x1.shape[0]
    t = torch.rand(batch_size, device=x1.device, dtype=x1.dtype)
    r = torch.rand(batch_size, device=x1.device, dtype=x1.dtype) * t  # 0 < r < t < 1
    x_t = x0 + t[:, None, None, None] * (x1 - x0)
    velocity = x1 - x0
    ones = torch.ones_like(t)

    # The JVP objective is numerically sensitive under AMP. Keep the derivative path in fp32
    # so the target tensor stays stable even on CUDA.
    with torch.autocast(device_type=x1.device.type, enabled=False):
        with torch.no_grad():
            _, derivative = jvp(
                lambda state, time: model(state, r, time),
                (x_t, t),
                (velocity, ones),
            )
        prediction = model(x_t, r, t)

    target = (velocity - (t - r)[:, None, None, None] * derivative).detach()
    loss = torch.mean((prediction - target) ** 2)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite MeanFlow loss encountered: {loss.item()}")
    return loss


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
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    distributed = world_size > 1

    if distributed:
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    output_dir = Path(config["out_dir"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train.log"
    if rank == 0 and log_path.exists():
        log_path.unlink()

    loader = build_loader(
        config["data_root"],
        config["image_size"],
        config["batch_size"],
        config["workers"],
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    model = MeanFlowUNet(base_channels=config["base_channels"]).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    use_amp = bool(config["amp"] and device.type == "cuda")
    scaler = build_scaler(device, use_amp)

    step = 0
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.module.load_state_dict(state["model"]) if hasattr(model, "module") else model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        if scaler is not None and "scaler" in state:
            scaler.load_state_dict(state["scaler"])
        step = state.get("step", 0)

    if rank == 0:
        print(f"device={device} distributed={distributed} rank={rank}/{world_size} images={len(loader.dataset):,} parameters={sum(p.numel() for p in model.parameters()):,}")
    model.train()

    epoch = 0
    while step < config["steps"]:
        if distributed and hasattr(loader.sampler, "set_epoch"):
            loader.sampler.set_epoch(epoch)

        for images, _ in loader:
            step += 1
            images = images.to(device, non_blocking=True)
            noise = torch.randn_like(images)
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

            if rank == 0 and step % config["log_every"] == 0:
                message = f"step={step}/{config['steps']} loss={loss.item():.5f}"
                print(message)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(message + "\n")

            if rank == 0 and step % config["sample_every"] == 0:
                save_samples(model, device, output_dir, step, config)

            if rank == 0 and step % config["ckpt_every"] == 0:
                state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                checkpoint = {"step": step, "model": state_dict, "optimizer": optimizer.state_dict(), "config": config}
                if scaler is not None:
                    checkpoint["scaler"] = scaler.state_dict()
                torch.save(checkpoint, output_dir / f"ckpt_step_{step}.pt")

            if step >= config["steps"]:
                break

        epoch += 1

    if rank == 0:
        final_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        final_checkpoint = {"step": step, "model": final_state_dict, "optimizer": optimizer.state_dict(), "config": config}
        if scaler is not None:
            final_checkpoint["scaler"] = scaler.state_dict()
        torch.save(final_checkpoint, output_dir / "model_final.pt")
        save_samples(model, device, output_dir, step, config)
        plot_loss_curve(log_path, output_dir / "loss_curve.png")

    if distributed:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    train(CONFIG, CONFIG["resume"])
