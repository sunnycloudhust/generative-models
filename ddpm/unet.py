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
        self.attention = nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
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


if __name__ == "__main__":
    batch_size = 2
    in_channels = 64
    out_channels = 8
    height = 32
    width = 32
    t_emb_dim = 128
    num_heads = 4

    x = torch.randn(batch_size, in_channels, height, width)
    time_steps = torch.tensor([10, 999])
    t_emb = get_time_embedding(time_steps, t_emb_dim)

    down_block = DownBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        t_emb_dim=t_emb_dim,
        down_sample=True,
        num_heads=num_heads,
    )

    print("x:", x.shape)
    print("t_emb:", t_emb.shape)

    out = down_block.resnet_conv_first(x)
    print("after resnet_conv_first:", out.shape)

    t_emb_out = down_block.t_emb_layers(t_emb)
    print("after t_emb_layers:", t_emb_out.shape)

    out = out + t_emb_out[:, :, None, None]
    print("after adding t_emb:", out.shape)

    out = down_block.resnet_conv_second(out)
    print("after resnet_conv_second:", out.shape)

    residual = down_block.residual_input_conv(x)
    print("residual:", residual.shape)

    out = out + residual
    print("after residual add:", out.shape)

    batch_size, channels, height, width = out.shape
    attn_input = down_block.attention_norm(out)
    print("after attention_norm:", attn_input.shape)

    attn_input = attn_input.reshape(batch_size, channels, height * width)
    print("after attention reshape:", attn_input.shape)

    attn_input = attn_input.transpose(1, 2)
    print("after attention transpose:", attn_input.shape)

    attn_output, _ = down_block.attention(attn_input, attn_input, attn_input)
    print("after attention:", attn_output.shape)

    attn_output = attn_output.transpose(1, 2).reshape(batch_size, channels, height, width)
    print("after attention restore:", attn_output.shape)

    out = out + attn_output
    print("after attention residual add:", out.shape)

    out = down_block.down_sample_conv(out)
    print("after down_sample_conv:", out.shape)

    final_out = down_block(x, t_emb)
    print("forward output:", final_out.shape)

