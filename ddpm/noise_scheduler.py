import torch
import torch.nn as nn

class LinearNoiseScheduler:
    def __init__(self, num_timesteps, beta_start, beta_end):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = torch.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = torch.sqrt(1.0 - self.alpha_cum_prod)
    
    def add_noise(self, original, noise, t):
        
        original_shape = original.shape
        batch_size = original.shape[0]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].reshape(batch_size)
    

if __name__ == "__main__":
    scheduler = LinearNoiseScheduler(
        num_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
    )

    original = torch.randn(2, 3, 32, 32)
    noise = torch.randn_like(original)
    t = torch.tensor([0, 999])

    noisy = scheduler.add_noise(original, noise, t)

    print("original shape:", original.shape)
    print("noise shape:", noise.shape)
    print("t:", t)
    print("noisy:", noisy)
