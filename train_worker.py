import os
import torch
from torch.optim.lr_scheduler import LambdaLR
from model import transformer
from train import train_process, batch, first_example
from train.dataset import get_dataloader, DataCtg

train_dataloader, lab_len, tok_len = get_dataloader(DataCtg.TRAIN, 32)
valid_dataloader, _, _ = get_dataloader(DataCtg.VAL, 32)

src_vocab_size = 2404
target_vocab_size = 25
target_padding = 24
src_padding = 2403
number_layers = 6


def train_worker(embed_dim, base_lr, lr_warmup, num_epochs, accum_iter):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = transformer.make_model(
        src_vocab=src_vocab_size,
        tgt_vocab=target_vocab_size,
        N=number_layers,
        d_model=embed_dim,
        d_ff=embed_dim * 4,
        h=8,
        dropout=0.2,
    )

    if os.path.exists('transformer_model.pth'):
        model=torch.load('transformer_model.pth')

    criterion = train_process.LabelSmoothing(
        target_vocab_size, padding_idx=target_padding, smoothing=0.1
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=base_lr, betas=(0.9, 0.98), eps=1e-9
    )

    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: train_process.rate(
            step, embed_dim, factor=1, warmup=lr_warmup
        ),
    )
    train_state = train_process.TrainState()

    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch} Training ====", flush=True)
        train_loss, train_state = train_process.run_epoch(
            (
                batch.Batch(
                    tgt=b[0], src=b[1], src_pad=src_padding, tgt_pad=target_padding
                )
                for b in train_dataloader
            ),
            model,
            train_process.SimpleLossCompute(model.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=accum_iter,
            train_state=train_state,
        )
        print(f'train loss: {train_loss}')

        model.eval()
        print(f"Epoch {epoch} Validation ====", flush=True)
        val_loss,_ = train_process.run_epoch(
            (batch.Batch(tgt=b[0], src=b[1], src_pad=src_padding, tgt_pad=target_padding) for b in valid_dataloader),
            model,
            train_process.SimpleLossCompute(model.generator, criterion),
            first_example.DummyOptimizer(),
            first_example.DummyScheduler(),
            mode="eval",
        )
        print(f'validation loss: {val_loss}')
        # torch.cuda.empty_cache()

    torch.save(model, 'transformer_model.pth')


if __name__ == "__main__":
    train_worker(
        embed_dim=256, base_lr=1, lr_warmup=4000, num_epochs=300, accum_iter=4
    )
