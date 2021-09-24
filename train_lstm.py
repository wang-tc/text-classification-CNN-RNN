import os
import sys


import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data.data_loading import CommentLabelDataset, collate
from train.lstm_train_func import train_lstm
from model.lstm_model import LSTMClassifier
from utils.func import load_json_file

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
VOCAB_SIZE = 39583
SEED = 0
BATCH_SIZE = 30
EMB_DIM = 64
LSTM_DIM = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device ='cpu'
print(device)

if __name__ == '__main__':
    data = pd.read_pickle('data/data_file/train_set.pkl')
    valid = pd.read_pickle('data/data_file/valid_set.pkl')
    valid = valid.head(1000)
    dataset = CommentLabelDataset(data)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    valid_set = CommentLabelDataset(valid)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    model = LSTMClassifier(vocab_size=VOCAB_SIZE,
                           embedding_dim=EMB_DIM,
                           hidden_dim=LSTM_DIM,
                           layer=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_lstm(
        lstm_model=model,
        optimizer=optimizer,
        num_epochs=2,
        device=device,
        train_loader=data_loader,
        valid_loader=valid_loader,
        eval_every=100,
        file_path='train/checkpoints/lstm',
        saving=True
    )
