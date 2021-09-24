# Libraries
import pandas as pd
import torch
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, f1_score, auc
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from time import time


# Evaluation Function
def evaluate_cnn(model, test_loader, version='title', threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    y_pred = []
    y_true = []
    y_prob = []

    model.eval()
    i = 0
    leng = len(test_loader)
    start = time()
    with torch.no_grad():
        for text, labels in test_loader:
            i += 1
            text = text.to(device)
            labels = labels.to(device)
            output = model(text)
            y_prob.extend(output.tolist())
            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
            sys.stdout.write("\r" + f'progress: {(i / leng): .1%}')
            sys.stdout.flush()
    end = time()
    sys.stdout.flush()
    sys.stdout.write("\r")
    print('model running time: ', end-start)
    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))
    lr_auc = roc_auc_score(y_true, y_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_prob)
    print('roc_auc_score:', lr_auc)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm, annot=True, ax=ax[0], cmap='Blues', fmt="d")

    ax[0].set_title('Confusion Matrix')

    ax[0].set_xlabel('Predicted Labels')
    ax[0].set_ylabel('True Labels')

    ax[0].xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax[0].yaxis.set_ticklabels(['FAKE', 'REAL'])

    ax[1].plot(lr_fpr, lr_tpr)
    # plt.xlabel('X Axis', axes=ax[1])
    # plt.ylabel('Y Axis', axes=ax[1])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC curve')

    ax[2].plot(lr_recall, lr_precision)
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_title('precision-recall curve')
    plt.show()


def evaluate_lstm(model, test_loader, version='title', threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    y_pred = []
    y_true = []
    y_prob = []

    model.eval()
    i = 0
    leng = len(test_loader)
    start = time()
    with torch.no_grad():
        for text, labels in test_loader:
            i += 1
            lens = (text != 0).sum(dim=1)
            text = text.to(device)
            labels = labels.to(device)
            output = model(text, lens)
            y_prob.extend(output.tolist())
            output = (output > threshold).int()
            y_pred.extend(output.tolist())
            y_true.extend(labels.tolist())
            sys.stdout.write("\r" + f'progress: {(i / leng): .1%}')
            sys.stdout.flush()
    end = time()
    sys.stdout.flush()
    sys.stdout.write("\r")
    print('model running time: ', end - start)

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[1, 0], digits=4))
    lr_auc = roc_auc_score(y_true, y_prob)
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_prob)
    print('roc_auc_score:', lr_auc)
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(cm, annot=True, ax=ax[0], cmap='Blues', fmt="d")

    ax[0].set_title('Confusion Matrix')

    ax[0].set_xlabel('Predicted Labels')
    ax[0].set_ylabel('True Labels')

    ax[0].xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax[0].yaxis.set_ticklabels(['FAKE', 'REAL'])

    ax[1].plot(lr_fpr, lr_tpr)
    # plt.xlabel('X Axis', axes=ax[1])
    # plt.ylabel('Y Axis', axes=ax[1])
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC curve')

    ax[2].plot(lr_recall, lr_precision)
    ax[2].set_xlabel('Recall')
    ax[2].set_ylabel('Precision')
    ax[2].set_title('precision-recall curve')
    plt.show()
