import math
from typing import Optional
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multiple parapllel attention mechanism, adapting self-attention
    and cross-attention in same time.
    """

    def __init__(self, embed_dim: int = 512, n_heads: int = 8) -> None:
        """
        Initialize multi-head attention.
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        # 512/8=64, each key, query, value will be of 64
        self.single_head_dim = int(self.embed_dim / self.n_heads)

        # key, query and value matrixes, shape is 64 x 64
        # Each head has different key, query and value matrixes.
        self.query_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.key_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.value_matrix = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # keep bias in output linear could enhence the flexi
        self.out = nn.Linear(self.n_heads * self.single_head_dim, self.embed_dim)

    def __get_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = (32 x 8 x 10 x 10)
        product = torch.matmul(query, key.transpose(-1, -2))

        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        product /= math.sqrt(self.single_head_dim)

        return torch.matmul(F.softmax(product, dim=-1), value)

    def __project(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        batch_size = key.size(0)
        seq_length = key.size(1)

        # query dimension can change in decoder during inference.
        # so we can't take general seq_length
        seq_length_query = query.size(1)

        q: torch.Tensor = self.query_matrix(query)
        k: torch.Tensor = self.key_matrix(key)
        v: torch.Tensor = self.value_matrix(value)

        # 32x10x512
        # batch_size x sequence_length x n_heads x single_head_dim = (32 x 10 x 8 x 64)
        # -> batch_size x n_heads x sequence_length x single_head_dim = (32 x 8 x 10 x 64)
        q = q.view(
            batch_size, seq_length_query, self.n_heads, self.single_head_dim
        ).transpose(1, 2)
        k = k.view(
            batch_size, seq_length, self.n_heads, self.single_head_dim
        ).transpose(1, 2)
        v = v.view(
            batch_size, seq_length, self.n_heads, self.single_head_dim
        ).transpose(1, 2)

        return q, k, v

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:  # batch_size x sequence_length x embedding_dim, 32 x 10 x 512
        """
        Args:
            query: query vector
            key: key vector
            value: value vector
            mask: mask for decoder
        Returns:
            torch.Tensor: vector from multihead attention
        """

        # project the vectors to n_heads lower dimension spaces
        # 32 x 10 x 512 -> 32 x 10 x 8 x 64
        q, k, v = self.__project(query, key, value)

        # acquire attention through softmax(QK^T/\sqrt{d_k})V
        attention = self.__get_attention(q, k, v, mask)

        # concatenated output
        # (32 x 8 x 10 x 64) -> (32 x 10 x 8 x 64) -> (32,10,512)
        concat = (
            attention.transpose(1, 2)
            .contiguous()
            .view(key.size(0), query.size(1), self.single_head_dim * self.n_heads)
        )

        return self.out(concat)


class TransformerBlock(nn.Module):
    """
    TransformerBlock represents a single block of the transformer encoder.

    The overall process of one block is: input -> multi-head self attention -> add & norm -> feed forward -> add & norm
    """

    def __init__(
        self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 8
    ) -> None:
        """Initializes the TransformerBlock


        Args:
            embed_dim: dimension of the embedding
            expansion_factor: factor which determines output dimension of linear layer
            n_heads: number of attention heads
        """
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.attention_norm = nn.LayerNorm(embed_dim)
        self.feed_norm = nn.LayerNorm(embed_dim)

        # FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )
        # enhence the generalization ability
        self.attention_dropout = nn.Dropout(0.2)
        self.feed_dropout = nn.Dropout(0.2)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Performs the forward pass for the transformer block.


        Args:
            query: query vector
            key: key vector
            value: value vector

        Returns:
            torch.Tensor: output of transformer block
        """
        attention_out = self.attention(query, key, value)

        # 32 x 10 x 512
        attention_norm_out = self.attention_dropout(
            self.attention_norm(attention_out + query)
        )

        # 32 x 10 x 512 -> 32 x 10 x 2048 -> 32 x 10 x 512
        feed_fwd_out = self.feed_forward(attention_norm_out)

        return self.feed_dropout(self.feed_norm(feed_fwd_out + attention_norm_out))


class DecoderBlock(nn.Module):
    """
    DecoderBlock include a multi-head self attention and a TransformerBlock (but use cross-attention).
    self-attention -> dropout -> TransformerBlock (cross-attention)
    """

    def __init__(
        self, embed_dim: int, expansion_factor: int = 4, n_heads: int = 8
    ) -> None:
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads

        """
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
           query: query vector from decoder
           key: key vector from encoder
           value: value vector from encoder
           mask: mask to be given for multi head attention

        Returns:
           torch.Tensor: output of transformer block
        """

        # we need to pass mask only to fst attention
        attention = self.attention(query, query, query, mask=mask)
        query = self.dropout(self.norm(attention + query))

        return self.transformer_block(query, key, value)
