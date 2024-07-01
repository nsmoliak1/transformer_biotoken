import torch
from torch import nn
import torch.nn.functional as F
from . import block
from . import embedding


class TransformerDecoder(nn.Module):
    """
    Decoder part of transformer. Responsible for encoding the input information.
    """

    def __init__(
        self,
        target_vocab_size,
        embed_dim,
        seq_len,
        num_layers=2,
        expansion_factor=4,
        n_heads=8,
    ) -> None:
        """
        Initialze TransformerDecoder

        Args:
            target_vocab_size: the vocabulary size of target code
            embed_dim: the embedding dimension
            seq_len: the max sequnce length of target
            num_layers: number of decoder layers
            expansion_factor: factor which determines number of linear layers in feed forward layer
            n_heads: number of heads in multihead attetion
        """
        super(TransformerDecoder, self).__init__()
        self.transformer_embedding = embedding.TransformerEmbedding(
            target_vocab_size, seq_len, embed_dim
        )

        self.layers = nn.ModuleList(
            [
                block.DecoderBlock(embed_dim, expansion_factor, n_heads)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_dim, target_vocab_size)

    def make_target_mask(self, target: torch.Tensor):

        batch_size, target_len, _ = target.shape
        # return the lower triangular part of matrix filled with ones
        target_mask = torch.tril(torch.ones((target_len, target_len))).expand(
            batch_size, 1, target_len, target_len
        )

        return target_mask

    def forward(self, x: torch.Tensor, enc_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input of decoder

        Return:
            torch.Tensor: output of decoder
        """
        x = self.transformer_embedding(x)
        mask = self.make_target_mask(x)

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, mask)

        return F.softmax(self.fc_out(x), dim=-1)
