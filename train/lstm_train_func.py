# https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0

# Training Function
# Libraries

import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

# Models
import tqdm
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Training
from model.lstm_model import LSTMClassifier
import torch.optim as optim
from utils.func import save_checkpoint, save_metrics, load_checkpoint, load_metrics
# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# import seaborn as sns

vocab_size = 39582
embedding_dim = 10
hidden_dim = 10


def train_lstm(
        lstm_model,
        optimizer,
        train_loader,
        valid_loader,
        num_epochs,
        eval_every,
        device,
        file_path,
        best_valid_loss=float("Inf"),
        criterion=nn.BCELoss(),
        saving=False

):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    lstm_model.train()
    for epoch in range(num_epochs):
        for text, labels in train_loader:
            lens = (text != 0).sum(dim=1)
            text = text.to(device)
            labels = labels.to(device)
            output = lstm_model(text, lens)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                lstm_model.eval()
                with torch.no_grad():
                    # validation loop
                    for text, labels in valid_loader:
                        lens = (text != 0).sum(dim=1)
                        text = text.to(device)
                        labels = labels.to(device)
                        output = lstm_model(text, lens)
                        loss = criterion(output, labels)
                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                lstm_model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if saving and best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', lstm_model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)

    if saving:
        save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    return valid_loss_list



