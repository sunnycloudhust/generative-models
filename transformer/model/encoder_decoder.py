import copy
import torch
import torch.nn as nn

from cross_attention import CrossAttention
from decoder import Decoder
from encoder import Encoder
from self_attention import SelfAttention

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, num_layer):
        super().__init__()
        self.num_layer = num_layer
        self.encoder_list = nn.ModuleList([
            copy.deepcopy(encoder) for _ in range(num_layer)])
        self.decoder_list = nn.ModuleList([
            copy.deepcopy(decoder) for _ in range(num_layer)])

    def forward(self, src, tgt, tgt_mask=None):
        encoder_output = src
        for encoder in self.encoder_list:
            encoder_output = encoder(encoder_output)

        decoder_output = tgt
        for decoder in self.decoder_list:
            decoder_output = decoder(
                decoder_output,
                k_state=encoder_output,
                v_state=encoder_output,
                mask=tgt_mask
            )

        return decoder_output


if __name__ == '__main__':
    batch_size = 256
    seq_len = 20
    d_model = 512
    d_ff = 1024
    num_head = 8
    num_layer = 2
    bias = True
    dropout = 0.1

    src = torch.randn(batch_size, seq_len, d_model)
    tgt = torch.randn(batch_size, seq_len, d_model)

    encoder = Encoder(SelfAttention(d_model, num_head, bias, dropout),
        d_model,
        d_ff,
        num_head,
        bias,
        dropout
    )
    decoder = Decoder(SelfAttention(d_model, num_head, bias, dropout),
        CrossAttention(d_model, num_head, bias, dropout),
        d_model,
        d_ff,
        num_head,
        bias,
        dropout
    )

    model = EncoderDecoder(encoder, decoder, num_layer)
    output = model(src, tgt)
    print(model)


