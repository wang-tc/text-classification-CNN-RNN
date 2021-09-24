import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data.data_loading import CommentLabelDataset
from train.cnn_train_func import Collate, train_cnn
from model.cnn_model import CNNClassifier
from dataclasses import dataclass

COLUMNS = ['out_size', 'stride', 'embedding_size', 'loss']
out_size_range = [24, 32, 40]
stride_range = [1, 2, 3]
embedding_range = [64, 96, 128]

VOCAB_SIZE = 39583
BATCH_SIZE = 30
SEED = 0

DATA_SIZE = 10000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    results = pd.DataFrame(columns=COLUMNS)
    i = 0
    data = pd.read_pickle('data/data_file/train_set.pkl')
    valid = pd.read_pickle('data/data_file/valid_set.pkl')
    data = data.head(DATA_SIZE)
    valid = valid.head(100)
    dataset = CommentLabelDataset(data)
    valid_set = CommentLabelDataset(valid)
    collate = Collate(322)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    for out in out_size_range:
        for stride in stride_range:
            for embedding_d in embedding_range:
                @dataclass
                class Parameters:
                    # Model parameters
                    embedding_size: int = embedding_d
                    out_size: int = out
                    stride: int = stride
                    # Preprocessing parameeters
                    seq_len: int = 322
                    num_words: int = 39583
                    # Training parameters
                    epochs: int = 1
                    batch_size: int = 30
                    learning_rate: float = 0.001
                model = CNNClassifier(Parameters).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                loss = train_cnn(
                    cnn_model=model,
                    optimizer=optimizer,
                    num_epochs=1,
                    device=device,
                    train_loader=data_loader,
                    valid_loader=valid_loader,
                    eval_every=DATA_SIZE//BATCH_SIZE+1,
                    file_path='train/checkpoints/cnn'
                )
                results.loc[i] = [out, stride, embedding_d, loss[-1]]
                i += 1
    results.to_csv('./tuning/cnn_tuning_result.csv', index=False)
