import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, bias=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.bias = bias
        self.dropout = nn.Dropout(p=dropout)
        self.d_head = self.d_model // self.num_head
        self.w_qkv = nn.linear(self.d_model, self.d_model * 3)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.scaling = math.sqrt(self.d_head)
    
    def forward(self, x):
        q,k,v = torch.split(self.linear(x), self.d_model, dim=1)        # Split into 3 matrices
        
        
        