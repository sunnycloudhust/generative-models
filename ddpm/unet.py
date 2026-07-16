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
        # Expected output shape = (batch_size, in_channels, height, width)
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        # Mapping the time embedding to the same dim as out_channels
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        ) # Expected output shape = (batch_size, out_channels, height, width)
        
        self.attention_norm = nn.GroupNorm(8, out_channels)
        self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                          stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb):
        out = self.resnet_conv_first(x)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(x)

        batch_size, channels, height, width = out.shape
        attn_input = self.attention_norm(out)
        attn_input = attn_input.reshape(batch_size, channels, height * width)
        attn_input = attn_input.transpose(1, 2)

        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, channels, height, width)
        out = out + attn_output

        return self.down_sample_conv(out)

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads):
        super().__init__()
        
        self.resnet_conv_first = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        
        self.t_emb_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            ),
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(t_emb_dim, out_channels)
            )
        ])
        
        self.resnet_conv_second = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                nn.GroupNorm(num_groups=8, num_channels=out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
        ])
        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        ])
    def forward(self, x, t_emb):
        out = x
        # --- 1. FIRST RESNET BLOCK ---
        resnet_input = out
        out = self.resnet_conv_first[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        # --- 2. ATTENTION BLOCK ---
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        # --- 3. SECOND RESNET BLOCK ---
        resnet_input = out
        out = self.resnet_conv_first[1](out)
        out = out + self.t_emb_layers[1](t_emb)[:, :, None, None]
        out = self.resnet_conv_second[1](out)
        out = out + self.residual_input_conv[1](resnet_input)
        
        return out
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, up_sample, num_heads):
        super().__init__()
        self.up_sample = up_sample
        
        self.resnet_conv_first = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.t_emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        )
        
        self.resnet_conv_second = nn.Sequential(
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.attention_norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
        self.residual_input_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.up_sample_conv = nn.ConvTranspose2d(
            in_channels // 2, in_channels // 2, kernel_size=4, stride=2, padding=1
        ) if self.up_sample else nn.Identity()
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        
        # Resnet block
        out = x
        resnet_input = out
        out = self.resnet_conv_first(out)
        out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)
        
        # Attention Block
        batch_size, channels, h, w = out.shape
        in_attn = out.reshape(batch_size, channels, h * w)
        in_attn = self.attention_norm(in_attn)
        in_attn = in_attn.transpose(1, 2)
        out_attn, _ = self.attention(in_attn, in_attn, in_attn)
        out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
        out = out + out_attn
        
        return out

