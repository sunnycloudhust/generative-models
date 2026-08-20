# Experiment of Flow Matching for CelebA Dataset

Note: For simplicity of this repo, some technical improvements have been discarded

A minimal flow matching implementation that learns a velocity field for transforming Gaussian noise into `64x64` CelebA face images. The model uses a U-Net, and the ODE is solved with the Euler method. It achieves FID score of 80.4752:`)))`

![Image generation trajectory with flow matching](assets/trajectory.png)

## Main Files

- `unet.py`: U-Net that predicts the time-dependent velocity field.
- `train.py`: trains the flow matching model and saves a checkpoint.
- `sample.py`: generates an image grid from a checkpoint.
- `eval/evaluate.py`: evaluates generated images with the Fréchet Inception Distance (FID).
- `config.py`: training and sampling hyperparameters.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparing the Dataset

Place `.jpg`, `.jpeg`, `.png`, or `.webp` images directly in the `data/` directory. The dataset loader expects a flat directory, center-crops each image, resizes it to `64x64`, and normalizes pixel values to `[-1, 1]`.

## Training

```bash
python train.py
```

The checkpoint and generated samples are saved in `runs/celeba_flow/`. The default settings use `10,000` training steps, a batch size of `64`, a learning rate of `2e-4`, and `100` Euler steps during sampling.

## Generating Images

After `runs/celeba_flow/model_final.pt` has been created, run:

```bash
python sample.py
```

The result is saved to `runs/celeba_flow/test_samples.png`.

## Evaluating FID

After training a model, evaluate it against the images in `data/`:

```bash
python eval/evaluate.py --num-samples 1000
```

The script generates the requested number of samples, extracts Inception v3 features from real and generated images, and prints the FID score. The first run downloads the pretrained Inception v3 weights. Use more samples for a more stable estimate.

You can override the default paths and sampling settings:

```bash
python eval/evaluate.py \
	--data-root ./data \
	--checkpoint ./runs/celeba_flow/model_final.pt \
	--num-samples 1000 \
	--batch-size 64 \
	--steps 100
```

## Notes

- You can configure the dataset, image size, number of steps, and device in `config.py`.
- CUDA is used automatically when available; otherwise, the code runs on the CPU.
- Checkpoints and generated outputs in `runs/` are ignored by Git. The image used in this README is stored separately at `assets/trajectory.png`.