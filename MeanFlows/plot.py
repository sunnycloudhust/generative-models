import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    epochs = []
    losses = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "loss=" not in line:
                continue
            try:
                epoch_value = None
                if "epoch=" in line:
                    epoch_part = line.split("epoch=")[-1].split()[0]
                    epoch_value = int(epoch_part.split("step=")[0].strip())

                left, loss_part = line.split("loss=")
                loss_value = float(loss_part)
                if epoch_value is not None:
                    epochs.append(epoch_value)
                else:
                    step_str = left.split("step=")[-1].split("/")[0]
                    epochs.append(int(step_str))
                losses.append(loss_value)
            except (ValueError, IndexError):
                continue

    return epochs, losses


def main():
    parser = argparse.ArgumentParser(description="Plot MeanFlow training loss curve.")
    parser.add_argument("--log", type=str, default="runs/celeba_meanflow/train.log", help="Path to the training log file.")
    parser.add_argument("--output", type=str, default="runs/celeba_meanflow/loss_curve.png", help="Output image path.")
    args = parser.parse_args()

    log_path = Path(args.log)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    steps, losses = parse_log(log_path)
    if len(steps) < 2:
        raise ValueError(f"Not enough valid loss entries in {log_path} to plot.")

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, linewidth=2, color="tab:blue")
    plt.title("MeanFlow training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
