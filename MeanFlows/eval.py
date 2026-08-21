import json
from pathlib import Path

import torch

from config import CONFIG
from data import build_loader
from train import meanflow_loss
from unet import MeanFlowUNet


def evaluate(config):
    device = torch.device(config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader = build_loader(
        config["data_root"], config["image_size"], config["batch_size"], config["workers"], split="val"
    )
    model = MeanFlowUNet(base_channels=config["base_channels"]).to(device)
    checkpoint = torch.load(config["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"] if "model" in checkpoint else checkpoint)
    model.eval()

    losses = []
    with torch.no_grad():
        for batch_index, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            noise = torch.randn_like(images)
            losses.append(meanflow_loss(model, noise, images).item())
            if batch_index + 1 >= config["eval_batches"]:
                break

    result = {
        "checkpoint": config["checkpoint"],
        "batches": len(losses),
        "meanflow_loss": sum(losses) / max(1, len(losses)),
    }
    output = Path(config["eval_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved {output}")
    return result


if __name__ == "__main__":
    evaluate(CONFIG)