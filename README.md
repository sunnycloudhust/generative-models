# Generative Models

A compact PyTorch repository for learning and experimenting with modern generative modeling approaches, including diffusion models, flow matching, score-based generative models, and Transformer-based sequence modeling.

## Overview

This project is organized as a collection of small, educational implementations rather than a production-ready framework. Each directory focuses on a different generative modeling idea and is intentionally structured to be readable and easy to modify.

## Repository Structure

- `ddpm/`
  - A DDPM-style diffusion implementation with a toy example and a small U-Net-based training pipeline.
- `flow-matching/`
  - A lightweight flow matching experiment folder with notes and model-building scaffolding.
- `score-based/`
  - Score-based generative modeling experiments, including a small training script and utilities.
- `transformer/`
  - An encoder-decoder Transformer implementation for English-to-Vietnamese translation.

## Getting Started

Create and activate a Python environment first, then install the required dependencies.

```bash
pip install torch torchvision
```

If you want to run the Transformer training pipeline with optional experiment logging, you may also install:

```bash
pip install wandb
```

## Running the Experiments

### DDPM toy example

```bash
cd ddpm/toy_example
python main.py
```

### Score-based model

```bash
python score-based/train.py
```

### Transformer translation experiment

```bash
cd transformer
python main.py
```

## Notes

- The repository is primarily for learning and experimentation.
- Some modules are still under development and may evolve as the code is extended.
- The project favors clarity and simple training loops over optimized production pipelines.

## License

This repository is provided for educational and research-oriented experimentation. Use the code accordingly.
