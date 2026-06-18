import torch 
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, bias=True, dropout=0.1):
        super().__init__()
        if d_model % num_head != 0:
            raise ValueError("d_model must be divisible by num_head")

        self.d_model = d_model
        self.bias = bias
        self.dropout = nn.Dropout(p=dropout)
        self.w_qkv = nn.Linear(self.d_model, self.d_model * 3, bias=bias)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias=bias)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.num_head = num_head
        self.d_head = self.d_model // self.num_head
        self.scaling = math.sqrt(self.d_head)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        q, k, v = torch.split(self.w_qkv(x), self.d_model, dim=-1)
        
        q = q.reshape(batch_size, seq_len, self.num_head, self.d_head).transpose(1, 2) #shape(b,num,seq,d_head)
        k = k.reshape(batch_size, seq_len, self.num_head, self.d_head).transpose(1, 2) #shape(b,num,seq,d_head)
        v = v.reshape(batch_size, seq_len, self.num_head, self.d_head).transpose(1, 2) #shape(b,num,seq,d_head)
        
        att_score = torch.matmul(q, k.transpose(-1, -2))
        att_score = att_score / self.scaling
        if mask is not None:
            att_score = att_score + mask

        att_weight = torch.softmax(att_score, dim=-1)
        att_weight = self.dropout(att_weight)
        
        out = torch.matmul(att_weight, v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, self.num_head * self.d_head)
        out = self.w_o(out)
        out = self.LayerNorm(out + x)
        return out
        
        
if __name__ == '__main__':
    vocab_size = 100
    d_model = 100
    seq_len = 20
    num_head = 10
    d_embed = 300
    bias = True
    batch_size = 256
    dropout = 0.1
    x = torch.randint(0, vocab_size, (batch_size, seq_len, d_model)).float() 
    attention = SelfAttention(d_model=d_model, num_head=num_head, bias=True, dropout=0.1)
    a = attention(x)
    print(a.shape)










        
        
        
        
        
        
        
        
        
        
