import os
import sys
import math
import numpy as np

import torch
import torch.nn as nn

from models.baseRNN import BaseRNN

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class EncoderRNN(BaseRNN):

    def __init__(self, vocab_size, max_len, hidden_size,
                 embedding_size, input_dropout_p, dropout_p, position_embedding,
                 pos_embedding, n_layers, bidirectional, rnn_cell, variable_lengths,
                 embedding, update_embedding, pos_add):
        super(EncoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p, n_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.s_rnn = rnn_cell
        if embedding is not None:
            self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.pos_add = pos_add
        if pos_add == 'cat':
            rnn_input_size = embedding_size*2
        else:
            rnn_input_size = embedding_size
        self.rnn = self.rnn_cell(rnn_input_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        self.position_embedding = position_embedding
        self.pos_embedding = pos_embedding

        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def sin_encoding(self, batch_size, max_len, input_lengths, d_model):
        pe = np.zeros((batch_size, max_len, d_model))
        for batch in range(batch_size):
            for pos in range(max_len):
                if input_lengths[batch] - pos >0:
                    for i in range(0, d_model, 2):
                        pe[batch, pos, i] = math.sin(
                                (input_lengths[batch]-pos)/(10000**(i/d_model)))
                        if i+1 == d_model:
                            break
                        pe[batch, pos, i+1] = math.cos(
                                (input_lengths[batch]-pos)/(10000**(i/d_model)))
                else:
                    for i in range(0, d_model, 2):
                        pe[batch, pos, i] = 0.0
                        if i+1 == d_model:
                            break
                        pe[batch, pos, i+1] = 0.0
        pos = torch.from_numpy(pe)
        if torch.cuda.is_available():
            pos = pos.type(torch.cuda.FloatTensor)
        return pos

    def length_encoding(self, batch_size, max_len, input_lengths):
        pe = []
        for batch in range(batch_size):
            p = []
            for pos in range(max_len):
                if input_lengths[batch] - pos >0:
                    p.append(input_lengths[batch]-pos)
                else:
                    p.append(0)
            pe.append(p)
        pos = torch.tensor(pe)
        if torch.cuda.is_available():
            pos = pos.cuda()
        posemb = self.pos_embedding(pos)
        return posemb

    def forward(self, input_var, input_lengths=None):
        batch_size = input_var.size(0)
        seq_len = input_var.size(1)

        if self.position_embedding == "sin":
            posemb = self.sin_encoding(
                batch_size, seq_len, input_lengths, self.embedding_size)
        if self.position_embedding == "length":
            posemb = self.length_encoding(batch_size, seq_len, input_lengths)

        embedded = self.embedding(input_var)
        if self.position_embedding is not None:
            if self.pos_add == 'cat':
                embedded = torch.cat((embedded, posemb), dim=2)
            elif self.pos_add == 'add':
                embedded += posemb
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(
                    embedded, input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden
