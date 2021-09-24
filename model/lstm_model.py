# https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
# https://towardsdatascience.com/lstm-text-classification-using-pytorch-2c6c657f8fc0
# https://gist.github.com/MikulasZelinka/9fce4ed47ae74fca454e88a39f8d911a

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            num_layers=layer,
                            batch_first=True
                            )

        self.drop = nn.Dropout(p=0.3)

        # The linear layer that maps from hidden state space to classification
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, sentence, sent_len):
        # print(torch.max(sentence.view(1,-1)))
        # try:
        #     assert((torch.max(sentence.view(1,-1)))<=self.vocab_size)
        # except:
        #     print(torch.max(sentence.view(1,-1)))
        # print(sentence)

        sent_embeds = self.word_embeddings(sentence)
        packed_input = pack_padded_sequence(sent_embeds, sent_len, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # get the last output
        output_ = output[range(len(output)), [i - 1 for i in sent_len], :self.hidden_dim]
        text_fea_d = self.drop(output_)
        text_fea_fc = self.fc(text_fea_d)
        text_fea_sq = torch.squeeze(text_fea_fc, 1)
        text_out = torch.sigmoid(text_fea_sq)
        # print('sent_len:', sent_len)
        # print('sentence:', sentence.shape)
        # print('embed:', sent_embeds.shape)
        # print('packed_input', packed_input.data.shape)
        # print('packed_output', packed_output.data.shape)
        # print('output', output.shape)
        # print('output_', output_.shape)
        #         print('text_fea', text_fea.shape)
        #         print('text_fea', text_fea.shape)

        return text_out
