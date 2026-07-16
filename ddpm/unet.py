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

    def forward(self, x, t_emb, return_skip=False):
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

        down_out = self.down_sample_conv(out)
        if return_skip:
            return down_out, out
        return down_out

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
        up_channels = in_channels - out_channels
        self.up_sample_conv = nn.ConvTranspose2d(
            up_channels, up_channels, kernel_size=4, stride=2, padding=1
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


class Unet(nn.Module):
    def __init__(
        self,
        im_channels=3,
        model_channels=(32, 64, 128, 256),
        t_emb_dim=128,
        num_heads=4,
    ):
        super().__init__()
        self.t_emb_dim = t_emb_dim

        self.conv_in = nn.Conv2d(im_channels, model_channels[0], kernel_size=3, stride=1, padding=1)

        self.downs = nn.ModuleList()
        in_channels = model_channels[0]
        for i, out_channels in enumerate(model_channels):
            down_sample = i != len(model_channels) - 1
            self.downs.append(
                DownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    t_emb_dim=t_emb_dim,
                    down_sample=down_sample,
                    num_heads=num_heads,
                )
            )
            in_channels = out_channels

        self.mid = MidBlock(
            in_channels=model_channels[-1],
            out_channels=model_channels[-1],
            t_emb_dim=t_emb_dim,
            num_heads=num_heads,
        )

        self.ups = nn.ModuleList()
        reversed_channels = list(reversed(model_channels))
        current_channels = reversed_channels[0]
        for i, skip_channels in enumerate(reversed_channels):
            up_sample = i != 0
            self.ups.append(
                UpBlock(
                    in_channels=current_channels + skip_channels,
                    out_channels=skip_channels,
                    t_emb_dim=t_emb_dim,
                    up_sample=up_sample,
                    num_heads=num_heads,
                )
            )
            current_channels = skip_channels

        self.norm_out = nn.GroupNorm(num_groups=8, num_channels=model_channels[0])
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(model_channels[0], im_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t):
        t_emb = get_time_embedding(t, self.t_emb_dim)

        out = self.conv_in(x)
        down_outputs = []
        for down in self.downs:
            out, skip = down(out, t_emb, return_skip=True)
            down_outputs.append(skip)

        out = self.mid(out, t_emb)

        for up in self.ups:
            out_down = down_outputs.pop()
            out = up(out, out_down, t_emb)

        out = self.norm_out(out)
        out = self.act_out(out)
        out = self.conv_out(out)
        return out


UNet = Unet
