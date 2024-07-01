from torch import nn
import torch
import math


class Embedding(nn.Module):
    """
    Embedding is used to convert words (or token) into vectors.
    """

    def __init__(self, vocab_size: int, embed_dim: int) -> None:
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out


class PositionalEmbedding(nn.Module):
    """
    In transformer, it will lose the positional information when processing
    words in input sentences in parallel. Thus positional embedding is indispensable.

    The final output embedding is $embedding + positional embedding$.
    If change the position of words, the positional embedding would not changed, therefore
    these two sentences will create different embedding sequences.
    """

    def __init__(self, max_seq_len: int, embed_model_dim: int) -> None:
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        # pe = torch.zeros(max_seq_len, embed_model_dim)
        # position = torch.arange(0, max_seq_len, dtype=torch.double).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, embed_model_dim, 2, dtype=torch.double)
        #     * (-math.log(1e4 / embed_model_dim))
        # )
        # pe[:, 0::2] = torch.sin(position * div_term)
        # pe[:, 1::2] = torch.cos(position * div_term)

        pe = torch.zeros(max_seq_len, self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(
                    pos / (10000 ** ((2 * (i + 1)) / self.embed_dim))
                )

        # [10, 512] -> [1, 10, 512]
        pe = pe.unsqueeze(0)

        # register buffer means the parameters in model, which should be saved and
        # restored in the state_dict, but not trained by the optimizer.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input vector
        Returns:
            x: output
        """

        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        # prevents the calculation of gradients for positional embedding
        # during forward propagation
        x = x + self.pe[:, :seq_len]
        # print("pe shape： ", self.pe.shape)
        return x


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size: int, max_seq_length: int, embed_dim: int) -> None:
        super(TransformerEmbedding, self).__init__()

        self.word_embedding = Embedding(vocab_size, embed_dim)
        self.positional_embedding = PositionalEmbedding(max_seq_length, embed_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.word_embedding(x)
        x = self.positional_embedding(x)
        return self.dropout(x)
