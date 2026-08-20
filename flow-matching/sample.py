import math
from pathlib import Path
import torch
from torchvision.utils import make_grid, save_image
from config import CONFIG
from train import sample
from unet import UNetVelocity

CHECKPOINT_PATH = "./runs/celeba_flow/model_final.pt"
OUTPUT_PATH = "./runs/celeba_flow/test_samples.png"
N_SAMPLES = 16
SAMPLE_STEPS = 100


def load_model(checkpoint_path, config, device):
    model = UNetVelocity(base_channels=config["base_channels"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    return model


@torch.no_grad()
def main():
    device_name = CONFIG["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_model(CHECKPOINT_PATH, CONFIG, device)
    x = sample(
        model=model,
        device=device,
        n_samples=N_SAMPLES,
        image_size=CONFIG["image_size"],
        steps=SAMPLE_STEPS,
    )
    grid = make_grid((x + 1) * 0.5, nrow=int(math.sqrt(N_SAMPLES)))
    save_image(grid, output_path)
    print(f"saved samples to {output_path}")


if __name__ == "__main__":
    main()
