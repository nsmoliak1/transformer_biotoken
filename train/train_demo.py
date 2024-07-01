import context

from model.transformer import Transformer
from train.dataset import get_dataloader, DataCtg


dataloader, lab_len, tok_len = get_dataloader(DataCtg.TRAIN, 16)


src_vocab_size = 1404
target_vocab_size = 25
num_layers = 6

model = Transformer(
    embed_dim=512,
    src_vocab_size=src_vocab_size,
    target_vocab_size=target_vocab_size,
    src_seq_length=tok_len,
    trg_seq_length=lab_len,
    num_layers=num_layers,
    expansion_factor=4,
    n_heads=8,
)
# print(model)

for label, token in dataloader:
    out = model(token, label)
    # detail of train process
    print(out.shape)
