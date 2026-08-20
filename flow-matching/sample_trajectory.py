import math
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from config import CONFIG
from sample import load_model


CHECKPOINT_PATH = "./runs/celeba_flow/model_final.pt"
OUTPUT_PATH = "./runs/celeba_flow/trajectory.png"
N_IMAGES = 2
SAMPLE_STEPS = 100
N_FRAMES = 8


@torch.no_grad()
def sample_trajectory(model, device, n_images, image_size, sample_steps, n_frames):
    model.eval()
    x = torch.randn(n_images, 3, image_size, image_size, device=device)
    frames = [x.clone()]
    save_steps = {
        round(i * sample_steps / (n_frames - 1))
        for i in range(1, n_frames)
    }

    dt = 1.0 / sample_steps
    for step in range(1, sample_steps + 1):
        t = torch.full((n_images,), (step - 1) * dt, device=device)
        x = x + model(x, t) * dt
        if step in save_steps:
            frames.append(x.clone())

    return torch.stack(frames, dim=1).flatten(0, 1).clamp(-1, 1)


def main():
    device_name = CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, CONFIG, device)
    trajectory = sample_trajectory(
        model=model,
        device=device,
        n_images=N_IMAGES,
        image_size=CONFIG["image_size"],
        sample_steps=SAMPLE_STEPS,
        n_frames=N_FRAMES,
    )
    grid = make_grid((trajectory + 1) * 0.5, nrow=N_FRAMES, padding=2)
    save_image(grid, output_path)
    print(f"saved trajectory to {output_path}")


if __name__ == "__main__":
    main()
