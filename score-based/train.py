from model import *
from utils import *
import os
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.
    Args:
        model: A PyTorch model instance that represents a time-dependent score-based model.
        x: A mini-batch of training data.    
        marginal_prob_std: A function that gives the standard deviation of the perturbation kernel.
        eps: A tolerance value for numerical stability.
    """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
    std = marginal_prob_std(random_t) # return sigma(t) for different images
    z = torch.randn_like(x)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3))) # use scaling sigma(t) squared 
    return loss


################ main training loop ################

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

n_epochs =  50
batch_size = 32 
lr=1e-4 
dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
optimizer = Adam(score_model.parameters(), lr=lr)
steps = len(data_loader)
log_interval = max(1, steps // 10)  # log every 10% of an epoch by default

checkpoint_path = 'checkpoint.pth'
start_epoch = 1

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state = checkpoint.get('model_state_dict', checkpoint)
    if hasattr(score_model, 'module'):
        score_model.module.load_state_dict(model_state)
    else:
        score_model.load_state_dict(model_state)
        
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Loaded checkpoint from '{checkpoint_path}', resuming at epoch {start_epoch}.")
    if start_epoch > n_epochs:
        print(f"Checkpoint already contains {checkpoint.get('epoch')} epochs; nothing to train.")
        exit(0)

for epoch in range(start_epoch, n_epochs + 1):
    avg_loss = 0.0
    num_items = 0
    for step, (x, y) in enumerate(data_loader, start=1):
        x = x.to(device)
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]

    # Print the averaged training loss for the epoch.
    epoch_loss = avg_loss / num_items
    print(f"Epoch {epoch}/{n_epochs} completed, avg loss: {epoch_loss:.6f}")

    # Update the checkpoint after each epoch of training.
    ckpt_state = score_model.module.state_dict() if hasattr(score_model, 'module') else score_model.state_dict()
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': ckpt_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to '{checkpoint_path}' after epoch {epoch}.")