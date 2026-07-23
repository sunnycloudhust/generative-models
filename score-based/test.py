import os
import torch
import torchvision.utils as vutils
from model import ScoreNet
from utils import diffusion_coeff_fn, marginal_prob_std_fn, device


def load_checkpoint(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def sample_images(model, batch_size=16, n_steps=100, save_path='samples.png', progress_save_path='samples_progress.png', n_progress_steps=8):
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=device)
    t = torch.linspace(1.0, 1e-3, n_steps, device=device)
    progress_indices = set(torch.linspace(0, n_steps - 1, steps=n_progress_steps, dtype=torch.long).cpu().tolist())
    progress_images = []

    for i in range(n_steps):
        ti = t[i]
        ti_next = t[i + 1] if i + 1 < n_steps else torch.tensor(1e-3, device=device)
        score = model(x, ti.expand(batch_size))
        g = diffusion_coeff_fn(ti)
        dt = ti_next - ti
        x = x + (g**2)[:, None, None, None] * score * dt
        if i + 1 < n_steps:
            noise_scale = torch.sqrt((ti_next - ti).abs())
            x = x + noise_scale * torch.randn_like(x)

        if i in progress_indices:
            x_vis = (x[[0]] + 1.0) / 2.0
            x_vis = torch.clamp(x_vis, 0.0, 1.0)
            progress_images.append(x_vis)

    x = (x + 1.0) / 2.0
    x = torch.clamp(x, 0.0, 1.0)
    vutils.save_image(x, save_path, nrow=4)

    if len(progress_images) > 0:
        progress_images = torch.cat(progress_images, dim=0)
        vutils.save_image(progress_images, progress_save_path, nrow=len(progress_images))
        print(f'Saved progress grid to {progress_save_path}')

    print(f'Saved samples to {save_path}')
    return x


if __name__ == '__main__':
    ckpt_path = 'ckpt.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"checkpoint not found '{ckpt_path}"
        )

    model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    model = load_checkpoint(model, ckpt_path)
    sample_images(model, batch_size=16, n_steps=100)
