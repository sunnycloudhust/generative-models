import torch
import torch.nn as nn
import math

class Preprocess(nn.Module):
    def __init__(self, vocab_size, d_model, d_embed, bias=True, dropout=0.1):
        super(Preprocess, self).__init__()
        self.d_model = d_model
        self.d_embed = d_embed
        self.bias = bias
        self.vocab_size = vocab_size
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(self.vocab_size, d_embed, padding_idx=0)
        self.linear = nn.Linear(d_embed, d_model)

    @staticmethod
    def sinusoidal_encoding(seq):
        position = torch.arange(0, seq.shape[1]).unsqueeze(1)
        pe = torch.zeros(seq.shape[1], d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        x = self.embedding(x) # return (batch_size, seq_len, d_embed)
        x = self.dropout(x)
        x = self.linear(x)
        positional_encoding = self.sinusoidal_encoding(x)
        return x + positional_encoding


if __name__ == '__main__':
    vocab_size = 100
    d_model = 100
    seq_len = 20
    d_embed = 300
    bias = True
    batch_size = 256
    dropout = 0.1
    model = Preprocess(vocab_size, d_model, d_embed, bias, dropout)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    a = model(x)
    print(a)





