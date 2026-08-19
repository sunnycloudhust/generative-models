import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------------------------------------------
# 1. Target distribution (2 moons)
# ------------------------------------------------------------------
def sample_two_moons(n, noise=0.06):
    n1 = n // 2
    n2 = n - n1

    theta1 = np.random.uniform(0, np.pi, n1)
    moon1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)

    theta2 = np.random.uniform(0, np.pi, n2)
    moon2 = np.stack([1 - np.cos(theta2), 1 - np.sin(theta2) - 0.5], axis=1)

    x = np.concatenate([moon1, moon2], axis=0)
    x += np.random.normal(scale=noise, size=x.shape)
    x = (x - np.array([0.5, -0.1])) * 1.5
    np.random.shuffle(x)
    return x.astype(np.float32)

# ------------------------------------------------------------------
# 2. Velocity field v_theta(x, t)
# ------------------------------------------------------------------

class VectorField(nn.Module):
    """MLP: input = [x (2-dimension vector), sin/cos embedding of t] -> output = 2-dimensional vector"""

    def __init__(self, dim=2, hidden=128, n_hidden_layers=3, n_freq=6):
        super().__init__()
        self.n_freq = n_freq
        time_dim = 2 * n_freq  

        layers = [nn.Linear(dim + time_dim, hidden), nn.SiLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, dim)]
        self.net = nn.Sequential(*layers)

    def time_embed(self, t):
        # t: (B,) trong [0, 1] -> sinusoidal features (B, 2*n_freq)
        freqs = 2 ** torch.arange(self.n_freq, device=t.device).float()  # (n_freq,)
        args = t[:, None] * freqs[None, :] * np.pi  # (B, n_freq)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x, t):
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        temb = self.time_embed(t)
        inp = torch.cat([x, temb], dim=-1)
        return self.net(inp)

# ------------------------------------------------------------------
# 3. Conditional Flow Matching loss
# ------------------------------------------------------------------

def train(model, n_steps=4000, batch_size=512, lr=2e-3, log_every=500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)
    losses = []

    for step in range(1, n_steps + 1):
        x1 = torch.from_numpy(sample_two_moons(batch_size)).to(DEVICE)  # data from the predefined target
        x0 = torch.randn_like(x1)                                       # noise
        t = torch.rand(batch_size, device=DEVICE)                       # t ~ U[0,1]

        t_ = t.view(-1, 1)
        x_t = (1 - t_) * x0 + t_ * x1        # linear interpolaton (conditional path)
        target_v = x1 - x0                    # conditional vector field

        pred_v = model(x_t, t)                  # predict the vector field
        loss = ((pred_v - target_v) ** 2).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

        if step % log_every == 0:
            avg = sum(losses[-log_every:]) / log_every
            print(f"step {step:5d}/{n_steps}  loss={avg:.4f}")

    return losses

# ------------------------------------------------------------------
# 4. Sampling: Euler solver from noise -> data
# ------------------------------------------------------------------
@torch.no_grad()
def sample(model, n_samples=1000, n_steps=100, return_traj=False):
    x = torch.randn(n_samples, 2, device=DEVICE)
    dt = 1.0 / n_steps
    traj = [x.clone().cpu().numpy()] if return_traj else None

    for i in range(n_steps):
        t = torch.full((n_samples,), i * dt, device=DEVICE)
        v = model(x, t)
        x = x + v * dt          # Euler step 
        if return_traj:
            traj.append(x.clone().cpu().numpy())

    x_np = x.cpu().numpy()
    return (x_np, traj) if return_traj else x_np


# ------------------------------------------------------------------
# 5. plotting
# ------------------------------------------------------------------
def plot_results(losses, model, save_path="fm_results.png"):
    x_gen, traj = sample(model, n_samples=1000, n_steps=100, return_traj=True)
    x_target = sample_two_moons(1000)
    x_source = np.random.randn(1000, 2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) loss curve
    ax = axes[0, 0]
    ax.plot(losses, lw=0.8)
    ax.set_title("CFM training loss")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")

    # (b) source distribution
    ax = axes[0, 1]
    ax.scatter(x_source[:, 0], x_source[:, 1], s=4, alpha=0.5, color="gray")
    ax.set_title("Nguồn: x0 ~ N(0, I)")
    ax.set_aspect("equal")

    # (c) target distribution
    ax = axes[0, 2]
    ax.scatter(x_target[:, 0], x_target[:, 1], s=4, alpha=0.5, color="tab:blue")
    ax.set_title("Đích thật: two moons")
    ax.set_aspect("equal")

    # (d) generated samples vs target
    ax = axes[1, 0]
    ax.scatter(x_target[:, 0], x_target[:, 1], s=4, alpha=0.3, color="tab:blue", label="target thật")
    ax.scatter(x_gen[:, 0], x_gen[:, 1], s=4, alpha=0.6, color="tab:red", label="sinh ra (model)")
    ax.set_title("Mẫu sinh ra sau khi tích phân ODE")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_aspect("equal")

    # (e) một vài quỹ đạo mẫu, từ t=0 đến t=1
    ax = axes[1, 1]
    traj = np.stack(traj)  # (n_steps+1, n_samples, 2)
    for i in range(0, 40):
        ax.plot(traj[:, i, 0], traj[:, i, 1], color="tab:green", alpha=0.35, lw=0.8)
    ax.scatter(traj[0, :40, 0], traj[0, :40, 1], s=10, color="gray", label="t=0 (noise)")
    ax.scatter(traj[-1, :40, 0], traj[-1, :40, 1], s=10, color="tab:red", label="t=1 (data)")
    ax.set_title("Quỹ đạo ODE học được (40 mẫu)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    # (f) vector field tại t=0.5, xem hướng học được ra sao
    ax = axes[1, 2]
    grid = np.linspace(-2.5, 2.5, 20)
    gx, gy = np.meshgrid(grid, grid)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)
    with torch.no_grad():
        pts_t = torch.from_numpy(pts).to(DEVICE)
        t_mid = torch.full((pts_t.shape[0],), 0.5, device=DEVICE)
        v_mid = model(pts_t, t_mid).cpu().numpy()
    ax.quiver(gx, gy, v_mid[:, 0].reshape(gx.shape), v_mid[:, 1].reshape(gx.shape),
              color="tab:purple", alpha=0.8, width=0.003)
    ax.scatter(x_target[:, 0], x_target[:, 1], s=2, alpha=0.15, color="tab:blue")
    ax.set_title("Trường vận tốc học được tại t=0.5")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    print(f"Đã lưu hình tại: {save_path}")


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
if __name__ == "__main__":
    model = VectorField().to(DEVICE)
    print(f"Device: {DEVICE}")
    print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")

    losses = train(model, n_steps=4000, batch_size=512, lr=2e-3)
    plot_results(losses, model, save_path="fm_results.png")