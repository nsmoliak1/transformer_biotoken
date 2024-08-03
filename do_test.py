import os
import torch
from model import transformer
from model import utils
from train.dataset import get_dataloader, DataCtg
from train.batch import Batch
from data.generate_dataset import decode_target
from tests import draw_imag, test_inference

src_vocab_size = 2404
target_vocab_size = 25
target_padding = 24
src_padding = 2403
number_layers = 6
embed_dim = 256

test_dataloader, lab_len, tok_len = get_dataloader(DataCtg.TRAIN, 1)


def inference_test():
    test_model = transformer.make_model(
        src_vocab=src_vocab_size,
        tgt_vocab=target_vocab_size,
        N=number_layers,
        d_model=embed_dim,
        d_ff=embed_dim * 4,
        h=8,
        dropout=0.2,
    )
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for _ in range(9):
        out = test_model.decode(
            memory, src_mask, ys, utils.subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


compare_list = ['VNR','RSM']

draw_imag.compare_sequences(compare_list)

# test_inference.run_tests()

