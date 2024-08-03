import torch.nn as nn
from . import utils
from . import structure


class Decoder(nn.Module):
    """
    Generic N layer decoder with masking.
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = utils.clones(layer, N)
        self.norm = structure.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward (defined below)
    """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = utils.clones(structure.SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        # self-attention -> residual -> cross-attention -> residual
        # -> feed forward -> residual
        m = memory
        # self-attention
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # cross-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # feed forward
        return self.sublayer[2](x, self.feed_forward)
