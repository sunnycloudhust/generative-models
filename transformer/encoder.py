import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from self_attention import SelfAttention
from cross_attention import CrossAttention

class Encoder(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff,
                num_head, bias=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.dropout = dropout
        self.bias = bias
        self.d_ff = d_ff
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model),
            nn.Dropout(p=self.dropout))
        self.LayerNorm = nn.LayerNorm(d_model)
        
    def forward(self, x, k_state, v_state, mask):
        x = self.self_attention(x, mask=mask)
        x = self.cross_attention(x, k_state, v_state)
        out1 = self.mlp(x)
        output = self.LayerNorm(out1 + x)
        return output
    
if __name__ == '__main__':
    vocab_size = 100000
    d_model = 512
    seq_len = 20
    d_ff = 1024
    num_head = 8
    d_embed = 256
    bias = True
    batch_size = 256
    mask = None
    dropout = 0.1
    x = torch.randint(0, vocab_size, (batch_size, seq_len, d_model)).float()
    k_state = torch.randint(0, vocab_size, (batch_size, seq_len, d_model)).float() 
    v_state = torch.randint(0, vocab_size, (batch_size, seq_len, d_model)).float()  
    self_attention = SelfAttention(d_model, num_head, bias, dropout)
    cross_attention = CrossAttention(d_model, num_head, bias, dropout)
    encoder = Encoder(self_attention, cross_attention, d_model, d_ff, num_head, bias, dropout)
    output = encoder(x, k_state, v_state, mask=None)
    print(output.shape)
    