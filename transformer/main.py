import argparse

from train import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train an English-to-Vietnamese Transformer.")
    parser.add_argument("--train-path", required=True, help="Path to .tsv/.txt/.csv/.jsonl parallel data.")
    parser.add_argument("--valid-path", default=None, help="Optional validation file.")
    parser.add_argument("--src-col", default="en", help="CSV/JSONL source column name.")
    parser.add_argument("--tgt-col", default="vi", help="CSV/JSONL target column name.")
    parser.add_argument("--valid-ratio", type=float, default=0.1)

    parser.add_argument("--max-src-len", type=int, default=80)
    parser.add_argument("--max-tgt-len", type=int, default=80)
    parser.add_argument("--src-vocab-size", type=int, default=30000)
    parser.add_argument("--tgt-vocab-size", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)

    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-embed", type=int, default=256)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--num-head", type=int, default=8)
    parser.add_argument("--num-layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="checkpoints/en_vi_transformer.pt")
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
