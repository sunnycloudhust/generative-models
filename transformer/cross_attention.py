import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    def __init__(self, d_model, num_head, bias=True, dropout=0.1):
        super().__init__()
        if d_model % num_head != 0:
            raise ValueError("d_model must be divisible by num_head")

        self.d_model = d_model
        self.num_head = num_head
        self.bias = bias
        self.dropout = nn.Dropout(p=dropout)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.d_head = self.d_model // self.num_head
        self.scaling = math.sqrt(self.d_head)
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
    def forward(self, x, k_state, v_state):
        batch_size, seq_len, d_model = x.shape
        if k_state.shape != (batch_size, seq_len, d_model) or v_state.shape != (batch_size, seq_len, d_model):
            raise ValueError("Input to the Decoder has different shape")
        k = self.w_k(k_state)
        v = self.w_k(v_state)
        q = self.w_q(x)
        
        k = k.reshape(batch_size, seq_len, self.num_head, self.d_head).transpose(1,2)
        q = q.reshape(batch_size, seq_len, self.num_head, self.d_head).transpose(1,2)
        v = v.reshape(batch_size, seq_len, self.num_head, self.d_head).transpose(1,2)

        cross_att_score = torch.matmul(q, k.transpose(-1, -2))
        cross_att_score /= self.scaling
        cross_att_score = F.softmax(cross_att_score, dim=-1)
        cross_att_score = self.dropout(cross_att_score)
        cross_att_score = torch.matmul(cross_att_score, v)
        out = cross_att_score.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, self.num_head * self.d_head)
        out = self.w_o(out)
        out = self.LayerNorm(out + x)
        return out    
        
if __name__ == '__main__':
    vocab_size = 100
    d_model = 500
    seq_len = 20
    num_head = 10
    d_embed = 300
    bias = True
    batch_size = 256
    dropout = 0.1
    x = torch.randint(0, vocab_size, (batch_size, seq_len, d_model)).float() 
    attention = CrossAttention(d_model=d_model, num_head=num_head, bias=True, dropout=0.1)
    k_state = torch.randint(0, vocab_size, (batch_size, seq_len, d_model)).float() 
    v_state = torch.randint(0, vocab_size, (batch_size, seq_len, d_model)).float() 

    a = attention(x, k_state=k_state, v_state=v_state)
    print(a.shape)







