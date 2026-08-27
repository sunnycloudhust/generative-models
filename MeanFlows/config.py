from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


CONFIG = {
    "data_root": PROJECT_ROOT / "data",
    "out_dir": "./runs/celeba_meanflow",
    "resume": None,
    "checkpoint": "./runs/celeba_meanflow/model_final.pt",
    "sample_output": "./runs/celeba_meanflow/samples.png",
    "image_size": 64,
    "batch_size": 1,
    "base_channels": 64,
    "steps": 10000,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "workers": 4,
    "amp": True,
    "log_every": 10,
    "sample_every": 100,
    "sample_steps": 50,
    "sample_n": 16,
    "ckpt_every": 1000,
    "seed": 42,
    "eval_batches": 100,
    "eval_output": "./runs/celeba_meanflow/eval.json",
}
