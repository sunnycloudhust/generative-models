import argparse
import json
from types import SimpleNamespace

from train import run_training


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train an English-to-Vietnamese Transformer.")
    parser.add_argument(
        "--config",
        default="config.json",
        help="Path to a JSON training config."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    print(f"loaded config: {args.config}")
    run_training(SimpleNamespace(**config))


if __name__ == "__main__":
    main()
