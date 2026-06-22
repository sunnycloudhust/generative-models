# Generative Models

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=sunnycloudhust_generative-models&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=sunnycloudhust_generative-models)

This repository contains small PyTorch implementations and experiments for generative modeling.

## Contents

- `ddpm/`
  - `noise_scheduler.py`: linear noise scheduler for DDPM-style diffusion models.
- `transformer/`
  - `preprocess.py`: token embedding projection with sinusoidal positional encoding.
  - `self_attention.py`: work-in-progress self-attention module.
  - `test.ipynb`: notebook for quick experiments.

## Setup

Create and activate a Python environment, then install PyTorch.

```bash
pip install torch
```

## Usage

Run individual modules directly while developing:

```bash
python transformer/preprocess.py
```

The code is currently organized as learning experiments, so APIs may change as the implementations are completed.
