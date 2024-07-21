from enum import Enum, auto
import os
import csv
import torch
import context
from torch.utils.data import Dataset, DataLoader, dataloader
from torch.nn.utils.rnn import pad_sequence

_max_tok_len = 18
_max_lab_len = 10


class DataCtg(Enum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class AminoDataSet(Dataset):
    def __init__(
        self,
        root_path,
        ctg=DataCtg.TRAIN,
    ):
        super(AminoDataSet, self).__init__()
        self.root_path = root_path
        self.ctg = ctg
        self.label_dict = dict()
        self.labels = list()
        self.tokens = list()

        with open(os.path.join(self.root_path, "target_vocab.csv")) as file:
            data_reader = csv.reader(file)
            for row in data_reader:
                self.label_dict[row[0]] = row[1]

        if self.ctg == DataCtg.TRAIN:
            file_path = os.path.join(self.root_path, "train_data.csv")
        elif self.ctg == DataCtg.VAL:
            file_path = os.path.join(self.root_path, "val_data.csv")
        elif self.ctg == DataCtg.TEST:
            file_path = os.path.join(self.root_path, "test_data.csv")
        else:
            raise ValueError("There only train, validation and test dataset.")

        with open(file_path, mode="r") as in_file:
            data_reader = csv.reader(in_file)
            for row in data_reader:
                self.labels.append(self.label2sequence(row[0]))
                self.tokens.append(self.sentence2token(row[1:]))

                # global _max_lab_len
                # if len(self.labels[-1]) > _max_lab_len:
                #     _max_lab_len = len(self.labels[-1])
                # global _max_tok_len
                # if len(self.tokens[-1]) > _max_tok_len:
                #     _max_tok_len = len(self.tokens[-1])
            self.length = len(self.labels)

    def __getitem__(self, index):
        label = [int(x) for x in self.labels[index]]
        token = [int(x) for x in self.tokens[index]]
        return label, token

    def __len__(self):
        return self.length

    def sentence2token(self, sentence):
        token = list()
        for word in sentence:
            if word == "<SOS>":
                token.append(2400)
            elif word == "<NOS>":
                token.append(2401)
            elif word == "<EOS>":
                token.append(2402)
            elif int(word) < 0:
                token.append(-int(word) + 1400)
            else:
                token.append(int(word))

        return token

    def label2sequence(self, label):
        result = list()
        result.append(self.label_dict["<SOS>"])
        for amino in label:
            result.append(self.label_dict[amino])
        result.append(self.label_dict["<EOS>"])

        return result


def collate_batch(batch):
    labels = [item[0] for item in batch]
    tokens = [item[1] for item in batch]
    global _max_tok_len
    global _max_lab_len

    for label in labels:
        if len(label) < _max_lab_len:
            label.extend([24] * int(_max_lab_len - len(label)))
    for token in tokens:
        if len(token) < _max_tok_len:
            token.extend([2403] * (_max_tok_len - len(token)))
    return torch.tensor(labels, dtype=torch.int), torch.tensor(tokens, dtype=torch.int)


def get_dataloader(ctg: DataCtg, batch_size=32):

    dataset = AminoDataSet(os.path.join(context.parent_dir, "data"), ctg)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )
    return dataloader, _max_lab_len, _max_tok_len


if __name__ == "__main__":
    dataloader, lab_len, tok_len = get_dataloader(DataCtg.TRAIN, 16)

    for label, token in dataloader:
        print("label: ", label, " token: ", token)

    print(lab_len, tok_len)
