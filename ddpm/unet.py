import torch 
import torch.nn as nn

def get_time_embedding(time_steps, t_emd_dim):
    # Expected time_steps.shape = (batch_size,)
    half_dim = t_emd_dim // 2
    if half_dim > 1:
        factor = torch.log(torch.tensor(10000.0, device=time_steps.device)) / (half_dim - 1)
        factor = torch.exp(torch.arange(half_dim, device=time_steps.device) * -factor)
    else:
        factor = torch.ones(half_dim, device=time_steps.device)

    time_steps = time_steps.float()
    embeddings = time_steps[:, None] * factor[None, :]
    t_emb = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    return t_emb     # Expected output.shape = (batch_size, t_emd_dim)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, down_sample, num_heads):
        super().__init__()
        self.down_sample = down_sample
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                          stride=2, padding=1) if self.down_sample else nn.Identity()





