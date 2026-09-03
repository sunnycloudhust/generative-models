import math
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from config import CONFIG
from unet import MeanFlowUNet


def sample(model, device, n_samples, image_size, steps, return_trajectory=False):
    was_training = model.training
    model.eval()
    x = torch.randn(n_samples, 3, image_size, image_size, device=device)
    trajectory = [x.clone()] if return_trajectory else None
    time_grid = torch.linspace(0.0, 1.0, steps + 1, device=device)
    with torch.no_grad():
        for start, end in zip(time_grid[:-1], time_grid[1:]):
            r = start.expand(n_samples)
            t = end.expand(n_samples)
            x = x + (end - start) * model(x, r, t)
            if return_trajectory:
                trajectory.append(x.clone())
    if was_training:
        model.train()
    if return_trajectory:
        return torch.stack(trajectory).clamp(-1.0, 1.0)
    return x.clamp(-1.0, 1.0)


def save_trajectory(trajectory, output, sample_index=0, columns=6):
    states = trajectory[:, sample_index]
    selected = torch.linspace(0, len(states) - 1, columns).round().long()
    states = states[selected]
    save_image(make_grid((states + 1.0) * 0.5, nrow=columns), output)


def main():
    device = torch.device(CONFIG.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = MeanFlowUNet(base_channels=CONFIG["base_channels"]).to(device)
    
    checkpoint = torch.load(CONFIG["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    images = sample(model, device, CONFIG["sample_n"], CONFIG["image_size"], CONFIG["sample_steps"])
    output = Path(CONFIG["sample_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    save_image(
        make_grid((images + 1.0) * 0.5, nrow=max(1, math.isqrt(CONFIG["sample_n"]))),
        output,
    )
    trajectory = sample(
        model,
        device,
        CONFIG["sample_n"],
        CONFIG["image_size"],
        CONFIG["sample_steps"],
        return_trajectory=True,
    )
    trajectory_output = Path(CONFIG["trajectory_output"])
    trajectory_output.parent.mkdir(parents=True, exist_ok=True)
    save_trajectory(trajectory, trajectory_output, columns=CONFIG["trajectory_columns"])
    print(f"saved {trajectory_output}")
    print(f"saved {output}")


if __name__ == "__main__":
    main()
