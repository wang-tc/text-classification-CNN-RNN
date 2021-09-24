import torch
from dataclasses import dataclass
import os
import sys
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Training
from model.cnn_model import CNNClassifier
import torch.optim as optim
from utils.func import save_checkpoint, save_metrics, load_checkpoint, load_metrics
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data.data_loading import CommentLabelDataset
from train.cnn_train_func import train_cnn, Collate


# VOCAB_SIZE = 39583
# SEED = 0
BATCH_SIZE = 30
# EMB_DIM = 64
#

# LSTM_DIM = 10
@dataclass
class Parameters:
    # Preprocessing parameeters
    seq_len: int = 322
    num_words: int = 39583

    # Model parameters
    embedding_size: int = 64
    out_size: int = 40
    stride: int = 2

    # Training parameters
    epochs: int = 1
    batch_size: int = 30
    learning_rate: float = 0.001


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    data = pd.read_pickle('data/data_file/train_set.pkl')
    valid = pd.read_pickle('data/data_file/train_set.pkl')
    valid = valid.head(1000)
    dataset = CommentLabelDataset(data)
    collate = Collate(Parameters.seq_len)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    valid_set = CommentLabelDataset(valid)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    model = CNNClassifier(Parameters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_cnn(
        cnn_model=model,
        optimizer=optimizer,
        num_epochs=2,
        device=device,
        train_loader=data_loader,
        valid_loader=valid_loader,
        eval_every=1000,
        file_path='train/checkpoints/cnn',
        saving=True
    )
