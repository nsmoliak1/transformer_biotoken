import torch
from torch import nn
from . import encoder
from . import decoder

print(torch.__version__)


class Transformer(nn.Module):
    """
    Transformer model
    """

    def __init__(
        self,
        embed_dim,
        src_vocab_size,
        target_vocab_size,
        src_seq_length,
        trg_seq_length,
        num_layers=2,
        expansion_factor=4,
        n_heads=8,
    ):
        """
        Initialize Transformer

        Args:
            embed_dim: the dimension of embedding vectors
            src_vocab_size: the size of source vocabulary
            target_vocab_size: the size of target vocabulary
            src_seq_length: the length of sequence of source
            trg_seq_length: the length of sequence of target
            num_layers: the number of layers
            expansion_factor: factor which determines number of linear layers in feed forward layer
            n_heads: number of heads in multihead attetion
        """
        super(Transformer, self).__init__()

        self.target_vocab_size = target_vocab_size
        self.encoder = encoder.TransformerEncoder(
            src_seq_length,
            src_vocab_size,
            embed_dim,
            num_layers,
            expansion_factor,
            n_heads,
        )
        self.decoder = decoder.TransformerDecoder(
            target_vocab_size,
            embed_dim,
            trg_seq_length,
            num_layers,
            expansion_factor,
            n_heads,
        )

    # TODO: adjust the length
    def transform(self, src: torch.Tensor, trg: torch.Tensor):
        enc_out = self.encoder(src)
        out_labels = []
        seq_len = src.shape[1]
        out = trg
        for _ in range(seq_len):
            out = self.decoder(out, enc_out)  # bs x seq_len x vocab+dim
            # taking the last token
            out = out[:, -1, :]

            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, dim=0)

        return out_labels

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(src)

        outputs = self.decoder(trg, enc_out)
        return outputs
