import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from noise_scheduler import LinearNoiseScheduler
from unet import Unet


DATA_DIR = "data"
OUTPUT_DIR = "runs/mnist_ddpm"

IMAGE_SIZE = 32
IM_CHANNELS = 1
BATCH_SIZE = 128
EPOCHS = 20
LR = 2e-4
NUM_WORKERS = 2
NUM_TIMESTEPS = 1000
BETA_START = 1e-4
BETA_END = 0.02

MODEL_CHANNELS = (32, 64, 128, 256)
T_EMB_DIM = 128
NUM_HEADS = 4

SAVE_INTERVAL = 5
SAMPLE_INTERVAL = 5
NUM_SAMPLES = 16
SEED = 42


def setup_logger(output_dir):
    log_path = output_dir / "train.log"
    logger = logging.getLogger("mnist_ddpm")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_config(logger, device):
    logger.info("Starting MNIST DDPM training")
    logger.info("device=%s", device)
    logger.info(
        "config image_size=%s batch_size=%s epochs=%s lr=%s num_timesteps=%s",
        IMAGE_SIZE,
        BATCH_SIZE,
        EPOCHS,
        LR,
        NUM_TIMESTEPS,
    )
    logger.info(
        "model im_channels=%s model_channels=%s t_emb_dim=%s num_heads=%s",
        IM_CHANNELS,
        MODEL_CHANNELS,
        T_EMB_DIM,
        NUM_HEADS,
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloader():
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }
    checkpoint_path = checkpoint_dir / f"ddpm_mnist_epoch_{epoch:04d}.pt"
    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, checkpoint_path)
    torch.save(checkpoint, latest_path)
    return checkpoint_path, latest_path


@torch.no_grad()
def sample(model, scheduler, device, sample_dir, epoch):
    model.eval()
    xt = torch.randn(NUM_SAMPLES, IM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

    for i in tqdm(reversed(range(NUM_TIMESTEPS)), total=NUM_TIMESTEPS, desc="sampling"):
        t = torch.full((NUM_SAMPLES,), i, device=device, dtype=torch.long)
        noise_pred = model(xt, t)
        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, i)

    xt = (xt.clamp(-1, 1) + 1) / 2
    sample_path = sample_dir / f"sample_epoch_{epoch:04d}.png"
    utils.save_image(xt, sample_path, nrow=4)
    model.train()
    return sample_path


def train():
    torch.manual_seed(SEED)

    device = get_device()
    output_dir = Path(OUTPUT_DIR)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)

    dataloader = build_dataloader()
    model = Unet(
        im_channels=IM_CHANNELS,
        model_channels=MODEL_CHANNELS,
        t_emb_dim=T_EMB_DIM,
        num_heads=NUM_HEADS,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END).to(device)

    log_config(logger, device)
    logger.info("num_batches_per_epoch=%s", len(dataloader))
    global_step = 0
    model.train()

    for epoch in range(1, EPOCHS + 1):
        progress = tqdm(dataloader, desc=f"epoch {epoch}/{EPOCHS}")
        running_loss = 0.0

        for batch_idx, (images, _) in enumerate(progress, start=1):
            images = images.to(device)
            noise = torch.randn_like(images)
            t = torch.randint(0, NUM_TIMESTEPS, (images.shape[0],), device=device)
            noisy_images = scheduler.add_noise(images, noise, t)

            noise_pred = model(noisy_images, t)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}", avg=f"{running_loss / batch_idx:.4f}")

        avg_loss = running_loss / len(dataloader)
        logger.info(
            "epoch=%s/%s global_step=%s avg_loss=%.6f",
            epoch,
            EPOCHS,
            global_step,
            avg_loss,
        )

        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            checkpoint_path, latest_path = save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir)
            logger.info("saved checkpoint=%s latest=%s", checkpoint_path, latest_path)

        if epoch % SAMPLE_INTERVAL == 0 or epoch == EPOCHS:
            sample_path = sample(model, scheduler, device, sample_dir, epoch)
            logger.info("saved sample=%s", sample_path)

    logger.info("Training finished")


if __name__ == "__main__":
    train()
