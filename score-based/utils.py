import torch 
import torch.nn as nn
import functools
import numpy as np 

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        # Expected return of the form (batch_size, embed_dim)        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def marginal_prob_std(t, sigma):
    """Compute standard deviation of $p_{0t}(x(t) | x(0))$.
    Args:    
        t: A vector of time steps.
        sigma: The $\sigma$ in our SDE.  
    
    Returns:
        The standard deviation.
    """    
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))   # cumulative sum of variances unitl time t

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

        Args:
            t: A vector of time steps.
            sigma: The $\sigma$ in our SDE.
        
        Returns:
            The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)    # return sigma(t)
  
sigma =  25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)