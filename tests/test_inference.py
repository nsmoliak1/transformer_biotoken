import os
import torch
from model import transformer
from model import utils
from train.dataset import get_dataloader, DataCtg
from train.batch import Batch
from data.generate_dataset import decode_target

src_vocab_size = 2404
target_vocab_size = 25
target_padding = 24
src_padding = 2403
number_layers = 6
embed_dim = 256

test_dataloader, lab_len, tok_len = get_dataloader(DataCtg.TRAIN, 1)


def run_tests():
    test_model = transformer.make_model(
        src_vocab=src_vocab_size,
        tgt_vocab=target_vocab_size,
        N=number_layers,
        d_model=embed_dim,
        d_ff=embed_dim * 4,
        h=8,
        dropout=0.2,
    )
    if os.path.exists("transformer_model.pth"):
        test_model = torch.load("transformer_model.pth")
    test_model.eval()

    data_list = [
        Batch(tgt=b[0], src=b[1], src_pad=src_padding, tgt_pad=target_padding)
        for b in test_dataloader
    ]

    for i in range(0, 10):
        # print(f"True label is: {data_list[i].tgt}")
        print(
            f"True aa sequence is: {decode_target(data_list[i].tgt.detach().numpy())}"
        )
        pred = transformer.greedy_decode(
            test_model,
            data_list[i].src,
            data_list[i].src_mask,
            max_len=10,
            start_symbol=0,
        )
        # print(f"Prediction label is: {pred}")
        print(f"Pred aa sequence is: {decode_target(pred.detach().numpy())}")
        print("======================")


# run_tests()
