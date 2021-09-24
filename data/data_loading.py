import torch
from torch.nn import Embedding, LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import pandas as pd

# constants
# NUM_WORDS = 10
SEED = 0
BATCH_SIZE = 3
# EMB_DIM = 2
# LSTM_DIM = 5

# for consistent results between runs
torch.manual_seed(SEED)


class CommentLabelDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return row['text_ids'], row['bool']

    def __len__(self):
        return len(self.data)


def collate(batch):
    sentences, labels = zip(*batch)
    # print(sentences)
    lens = list(map(len, sentences))
    sents_tensor = torch.zeros(len(batch), max(lens)).int()
    for i, sent in enumerate(sentences):
        sents_tensor[i, 0:lens[i]] = torch.tensor(sent)

    labels = torch.tensor(labels, dtype=torch.float32)
    return sents_tensor, labels


