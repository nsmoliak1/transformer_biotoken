import torch
from torch import nn
from . import embedding
from . import block


class TransformerEncoder(nn.Module):
    """
    The encoder part of transformer. Responsible for encoding the input information.
    """

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        embed_dim: int,
        num_layrs: int = 2,
        expansion_factor: int = 4,
        n_heads: int = 8,
    ) -> None:
        """
        Initializes the TransformerEncoder

        Args:
            seq_len: length of input sequence
            embed_dim: dimension of embedding
            num_layers: number of encoder layers
            expansion_factor: factor which determines number of linear layers in feed forward layer
            n_heads: number of heads in multihead attetion
        """
        super(TransformerEncoder, self).__init__()
        self.transformer_embedding = embedding.TransformerEmbedding(
            vocab_size, seq_len, embed_dim
        )

        self.layers = nn.ModuleList(
            [
                block.TransformerBlock(embed_dim, expansion_factor, n_heads)
                for _ in range(num_layrs)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass for the transformer block.

        Args:
            x: input of encoder

        Returns:
            torch.Tensor: output of the encoder
        """
        out = self.transformer_embedding(x)
        for layer in self.layers:
            out = layer(out, out, out)

        return out
