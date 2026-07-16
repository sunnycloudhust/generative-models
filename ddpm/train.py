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
    torch.save(checkpoint, checkpoint_dir / f"ddpm_mnist_epoch_{epoch:04d}.pt")
    torch.save(checkpoint, checkpoint_dir / "latest.pt")


@torch.no_grad()
def sample(model, scheduler, device, sample_dir, epoch):
    model.eval()
    xt = torch.randn(NUM_SAMPLES, IM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device=device)

    for i in tqdm(reversed(range(NUM_TIMESTEPS)), total=NUM_TIMESTEPS, desc="sampling"):
        t = torch.full((NUM_SAMPLES,), i, device=device, dtype=torch.long)
        noise_pred = model(xt, t)
        xt, _ = scheduler.sample_prev_timestep(xt, noise_pred, i)

    xt = (xt.clamp(-1, 1) + 1) / 2
    utils.save_image(xt, sample_dir / f"sample_epoch_{epoch:04d}.png", nrow=4)
    model.train()


def train():
    torch.manual_seed(SEED)

    device = get_device()
    output_dir = Path(OUTPUT_DIR)
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    dataloader = build_dataloader()
    model = Unet(
        im_channels=IM_CHANNELS,
        model_channels=MODEL_CHANNELS,
        t_emb_dim=T_EMB_DIM,
        num_heads=NUM_HEADS,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = LinearNoiseScheduler(NUM_TIMESTEPS, BETA_START, BETA_END).to(device)

    print(f"Training on {device}")
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

        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            save_checkpoint(model, optimizer, epoch, global_step, checkpoint_dir)

        if epoch % SAMPLE_INTERVAL == 0 or epoch == EPOCHS:
            sample(model, scheduler, device, sample_dir, epoch)


if __name__ == "__main__":
    train()
