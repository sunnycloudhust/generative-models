import torch
import torch.nn as nn

from cross_attention import CrossAttention
from decoder import Decoder
from encoder import Encoder
from encoder_decoder import EncoderDecoder
from preprocess import Preprocess
from self_attention import SelfAttention

class Transformer(nn.Module):
    def __init__(self, src_preprocess, tgt_preprocess, encoder_decoder, d_model, tgt_vocab_size, bias=True):
        super().__init__()
        self.src_preprocess = src_preprocess
        self.tgt_preprocess = tgt_preprocess
        self.encoder_decoder = encoder_decoder
        self.output_linear = nn.Linear(d_model, tgt_vocab_size, bias=bias)

    @staticmethod
    def make_tgt_mask(tgt): #create mask for masked self attention has shape (tgt, tgt)
        seq_len = tgt.shape[1]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=tgt.device),
            diagonal=1
        )
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, src, tgt, tgt_mask=None):
        if tgt_mask is None:
            tgt_mask = self.make_tgt_mask(tgt)

        src = self.src_preprocess(src)
        tgt = self.tgt_preprocess(tgt)
        output = self.encoder_decoder(src, tgt, tgt_mask)
        logits = self.output_linear(output)
        return logits


if __name__ == '__main__':
    batch_size = 256
    src_seq_len = 20
    tgt_seq_len = 16
    src_vocab_size = 1000
    tgt_vocab_size = 1200
    d_model = 512
    d_embed = 256
    d_ff = 1024
    num_head = 8
    num_layer = 6
    bias = True
    dropout = 0.1

    src_preprocess = Preprocess(src_vocab_size, d_model, d_embed, bias, dropout)
    tgt_preprocess = Preprocess(tgt_vocab_size, d_model, d_embed, bias, dropout)

    encoder = Encoder(
        SelfAttention(d_model, num_head, bias, dropout),
        d_model,
        d_ff,
        num_head,
        bias,
        dropout
    )
    decoder = Decoder(
        SelfAttention(d_model, num_head, bias, dropout),
        CrossAttention(d_model, num_head, bias, dropout),
        d_model,
        d_ff,
        num_head,
        bias,
        dropout
    )
    encoder_decoder = EncoderDecoder(encoder, decoder, num_layer)

    model = Transformer(
        src_preprocess,
        tgt_preprocess,
        encoder_decoder,
        d_model,
        tgt_vocab_size,
        bias
    )

    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    logits = model(src, tgt)
    print(logits.shape)
