import torch.nn as nn
from . import utils
from . import structure


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers
    """

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = utils.clones(layer, N)
        self.norm = structure.LayerNorm(layer.size)

    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """
    Encoder is made up of self-attention and feed forward (defined below)
    """

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = utils.clones(structure.SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        # self-attn -> residual -> feed forward -> residual
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
