from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchvision.utils import save_image

from model import *
from unet import *
from dataset import *


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
SAMPLE_DIR = Path(__file__).resolve().parent / "samples"
CHECKPOINT_DIR.mkdir(exist_ok=True)
SAMPLE_DIR.mkdir(exist_ok=True)

SAVE_EVERY = 10
SAMPLE_EVERY = 20
EPOCHS = 500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, DEVICE)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


def save_checkpoint(epoch, model, optimizer, loss):
    checkpoint_path = CHECKPOINT_DIR / f"ddpm_epoch_{epoch:03d}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        checkpoint_path,
    )
    print(f"Saved checkpoint to {checkpoint_path}")


@torch.no_grad()
def sample_images(model, epoch):
    model.eval()
    sample_batch = torch.randn(4, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    for t in reversed(range(T)):
        t_batch = torch.full((sample_batch.shape[0],), t, device=DEVICE, dtype=torch.long)
        predicted_noise = model(sample_batch, t_batch)

        sqrt_alpha_t = sqrt_alphas_cumprod[t].to(DEVICE)
        sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[t].to(DEVICE)

        pred_x0 = (sample_batch - sqrt_one_minus_alpha_t * predicted_noise) / sqrt_alpha_t.clamp_min(1e-8)
        sample_batch = pred_x0

    sample_batch = (sample_batch + 1) / 2
    sample_batch = sample_batch.clamp(0, 1)
    sample_path = SAMPLE_DIR / f"sample_epoch_{epoch:03d}.png"
    save_image(sample_batch, sample_path, nrow=2)
    print(f"Saved sample image to {sample_path}")
    model.train()


model.to(DEVICE)
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        t = torch.randint(0, T, (BATCH_SIZE,), device=DEVICE).long()
        loss = get_loss(model, batch, t)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / max(1, len(dataloader))

    if epoch % SAVE_EVERY == 0:
        save_checkpoint(epoch, model, optimizer, avg_loss)

    if epoch % SAMPLE_EVERY == 0:
        sample_images(model, epoch)

    if epoch % 5 == 0:
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.6f}")
