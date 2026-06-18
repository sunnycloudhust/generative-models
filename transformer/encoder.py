import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from self_attention import SelfAttention

class Encoder(nn.Module):
    def __init__(self, self_attention, d_model, d_ff, num_head, 
                bias=True, dropout=0.1):
        super().__init__()
        self.self_attention = self_attention
        self.d_model = d_model
        self.dropout = dropout
        self.bias = bias
        self.d_ff = d_ff
        self.num_head = num_head
        
    def forward(self, x, mask, )
