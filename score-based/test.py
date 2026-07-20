import os
import torch
import torchvision.utils as vutils
from model import ScoreNet
from utils import marginal_prob_std_fn, device


def load_checkpoint(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location=device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def sample_images(model, batch_size=16, n_steps=100, step_size=2e-5, save_path='samples.png'):
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28, device=device)

    for step in range(n_steps):
        t = torch.full((batch_size,), 1.0 - step / max(1, n_steps), device=device)
        score = model(x, t)
        std = marginal_prob_std_fn(t)[:, None, None, None]
        x = x + (step_size / (std ** 2)) * score
        x = x + torch.sqrt(2.0 * step_size) * torch.randn_like(x)

    x = (x + 1.0) / 2.0
    vutils.save_image(x, save_path, nrow=4)
    print(f'Saved samples to {save_path}')
    return x


if __name__ == '__main__':
    ckpt_path = 'ckpt.pth'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"checkpoint not found '{ckpt_path}'. Hãy chạy train.py trước rồi mới chạy test.py."
        )

    model = ScoreNet(marginal_prob_std=marginal_prob_std_fn).to(device)
    model = load_checkpoint(model, ckpt_path)
    sample_images(model, batch_size=16, n_steps=100, step_size=2e-5)
