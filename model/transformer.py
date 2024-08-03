import copy
import torch
import torch.nn as nn
from . import structure, encoder, decoder, utils


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    # Helper: Construct a model from hyperparameters.
    c = copy.deepcopy
    attn = structure.MultiHeadedAttention(h, d_model)
    ff = structure.PositionwiseFeedForward(d_model, d_ff, dropout)
    position = structure.PositionalEncoding(d_model, dropout)
    model = structure.EncoderDecoder(
        encoder.Encoder(encoder.EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder.Decoder(
            decoder.DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N
        ),
        nn.Sequential(structure.Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(structure.Embeddings(d_model, tgt_vocab), c(position)),
        structure.Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, utils.subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
