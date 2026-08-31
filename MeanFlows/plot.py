import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    values = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "loss=" not in line:
                continue
            try:
                if "epoch=" in line:
                    left, loss_part = line.split("loss=")
                    loss_value = float(loss_part)
                    epoch_part = left.split("epoch=")[-1].split()[0]
                    epoch_value = int(epoch_part)
                    values.append((epoch_value, loss_value))
                else:
                    left, loss_part = line.split("loss=")
                    loss_value = float(loss_part)
                    step_str = left.split("step=")[-1].split("/")[0]
                    values.append((int(step_str), loss_value))
            except (ValueError, IndexError):
                continue
    return [x for x, _ in values], [y for _, y in values]


def plot_loss_curve(log_path, output_path):
    log_path = Path(log_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    x, y = parse_log(log_path)
    if len(x) < 2:
        raise ValueError(f"Not enough valid loss entries in {log_path} to plot.")

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, linewidth=2, color="tab:blue")
    plt.title("MeanFlow training loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MeanFlow training loss curve.")
    parser.add_argument("--log", type=str, default="runs/celeba_meanflow/train.log", help="Path to the training log file.")
    parser.add_argument("--output", type=str, default="runs/celeba_meanflow/loss_curve.png", help="Output image path.")
    args = parser.parse_args()
    plot_loss_curve(args.log, args.output)


if __name__ == "__main__":
    main()
