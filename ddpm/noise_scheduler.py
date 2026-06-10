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
        batch_size = original_shape[0]
        
        self.sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod[t].view(batch_size, 1, 1, 1)
        self.sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod[t].view(batch_size, 1, 1, 1)
        noisy = self.sqrt_alpha_cum_prod * original + self.sqrt_one_minus_alpha_cum_prod * noise
        return noisy
    
    def sample_noise(self, 