import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import Decoder
from encoder import Encoder
class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, num_layer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_layer = num_layer
    
    def forward(self, x):
        
        