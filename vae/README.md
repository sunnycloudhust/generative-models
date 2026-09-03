# MNIST Variational Autoencoder

A compact PyTorch implementation of a variational autoencoder (VAE) trained on MNIST. The project trains an encoder-decoder model, saves reconstruction comparisons, generates random samples, and exports the trained weights.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- torchvision 0.15+

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Train

From this directory, run:

```bash
python train_vae.py
```

MNIST is downloaded to `data/` on the first run. Outputs are written to `outputs/`:

- `reconstruction_epoch_*.png`: original images and their reconstructions
- `samples_epoch_*.png`: images sampled from the latent space
- `vae_mnist.pt`: trained model weights

## Options

```bash
python train_vae.py \
  --epochs 10 \
  --batch-size 128 \
  --latent-dim 20 \
  --learning-rate 0.001 \
  --data-dir data \
  --output-dir outputs \
  --seed 42
```

Use `python train_vae.py --help` to see all available options.

## Project structure

- `model.py`: VAE encoder, latent reparameterization, and decoder
- `data.py`: MNIST data loader
- `losses.py`: reconstruction and KL-divergence losses
- `trainer.py`: training loop for one epoch
- `visualize.py`: reconstruction and sample image export
- `train_vae.py`: command-line training entry point
