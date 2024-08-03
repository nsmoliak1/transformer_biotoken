# import context
from model import utils


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """

    def __init__(self, src, tgt=None, src_pad=2, tgt_pad=2):
        self.src = src
        self.src_mask = (src != src_pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, tgt_pad)
            self.ntokens = (self.tgt_y != tgt_pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        # Create a mask to hide padding and future words.
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & utils.subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask
