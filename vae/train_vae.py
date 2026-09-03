"""Train a variational autoencoder on MNIST."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import optim

from data import create_mnist_loader
from model import VAE
from trainer import train_epoch
from visualize import save_reconstructions, save_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_mnist_loader(args.data_dir, args.batch_size)

    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Training on {device} with {len(train_loader.dataset):,} images")

    for epoch in range(1, args.epochs + 1):
        loss, reconstruction_loss, kl_loss = train_epoch(
            model, train_loader, optimizer, device
        )
        save_reconstructions(model, train_loader, device, args.output_dir, epoch)
        save_samples(model, device, args.output_dir, epoch)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"loss={loss:.4f} recon={reconstruction_loss:.4f} kl={kl_loss:.4f}"
        )

    torch.save(model.state_dict(), args.output_dir / "vae_mnist.pt")
    print(f"Saved model and images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()