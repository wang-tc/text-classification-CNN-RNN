from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

# Models
import tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils.func import save_checkpoint, save_metrics, load_checkpoint, load_metrics
# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# import seaborn as sns

@dataclass
class Parameters:
    # Preprocessing parameeters
    seq_len: int = 322
    num_words: int = 39583

    # Model parameters
    embedding_size: int = 64
    out_size: int = 32
    stride: int = 2

    # Training parameters
    epochs: int = 1
    batch_size: int = 30
    learning_rate: float = 0.001


class Collate:
    def __init__(self, seq_len):
        self.seq_len = seq_len

    def __call__(self, batch):
        sentences, labels = zip(*batch)
        # print(sentences)
        lens = list(map(len, sentences))
        sents_tensor = torch.zeros(len(batch), self.seq_len).int()
        for i, sent in enumerate(sentences):
            sents_tensor[i, 0:lens[i]] = torch.tensor(sent)
        labels = torch.tensor(labels, dtype=torch.float32)
        return sents_tensor, labels


def train_cnn(
        cnn_model,
        optimizer,
        train_loader,
        valid_loader,
        num_epochs,
        eval_every,
        device,
        file_path,
        best_valid_loss=float("Inf"),
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
    cnn_model.train()
    for epoch in range(num_epochs):
        for text, labels in train_loader:
            text = text.to(device)
            labels = labels.to(device)
            output = cnn_model(text)
            loss = F.binary_cross_entropy(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # update running values
            running_loss += loss.item()
            global_step += 1
            # evaluation step
            if global_step % eval_every == 0:
                cnn_model.eval()
                with torch.no_grad():
                    # validation loop
                    for text, labels in valid_loader:
                        text = text.to(device)
                        labels = labels.to(device)
                        output = cnn_model(text)
                        loss = F.binary_cross_entropy(output, labels)
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
                cnn_model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if saving and best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(file_path + '/model.pt', cnn_model, optimizer, best_valid_loss)
                    save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    if saving:
        save_metrics(file_path + '/metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
    return valid_loss_list
