import pandas as pd
import torch
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data.data_loading import CommentLabelDataset, collate
from train.lstm_train_func import train_lstm
from model.lstm_model import LSTMClassifier

COLUMNS = ['lstm_layer', 'embedding dimension', 'hidden_dimension', 'loss']
LAYER_RANGE = [1, 2, 3]
EMBEDDING_RANGE = [64, 96, 128]
HIDDEN_RANGE = [10, 15, 20]

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
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    for layer in LAYER_RANGE:
        for embedding_d in EMBEDDING_RANGE:
            for hidden_d in HIDDEN_RANGE:
                model = LSTMClassifier(
                    vocab_size=VOCAB_SIZE,
                    embedding_dim=embedding_d,
                    hidden_dim=hidden_d,
                    layer=layer
                ).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                loss = train_lstm(
                    lstm_model=model,
                    optimizer=optimizer,
                    num_epochs=1,
                    device=device,
                    train_loader=data_loader,
                    valid_loader=valid_loader,
                    eval_every=DATA_SIZE//BATCH_SIZE+1,
                    file_path='train/checkpoints/lstm'
                )
                results.loc[i] = [layer, embedding_d, hidden_d, loss[-1]]
                i += 1
    results.to_csv('./tuning/lstm_tuning_result.csv', index=False)