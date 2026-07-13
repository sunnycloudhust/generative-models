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
    embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

    if t_emd_dim % 2 == 1:
        embeddings = torch.nn.functional.pad(embeddings, (0, 1))

    return embeddings       # Expected output.shape = (batch_size, t_emd_dim)







