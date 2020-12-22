import os
import sys
import random
import math
import copy

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.attention import Attention
from models.baseRNN import BaseRNN

# GPU check
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_ENCODER_OUTPUTS = 'encoder_outputs'

    def __init__(self, vocab, vocab_size, max_len, hidden_size, embedding_size,
            sos_id, eos_id, input_dropout_p, dropout_p, position_embedding,
            pos_embedding, n_layers, bidirectional, rnn_cell, use_attention, pos_add):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.s_rnn = rnn_cell
        self.init_input = None
        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.output_size, embedding_size)
        self.pos_embedding = pos_embedding
        self.position_embedding = position_embedding
        if position_embedding is not None:
            self.insert_tok = self.insert_token(vocab)
        self.pos_add = pos_add
        if pos_add == 'cat':
            rnn_input_size = embedding_size*2
        else:
            rnn_input_size = embedding_size
        self.rnn = self.rnn_cell(rnn_input_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        if use_attention:
            self.attention = Attention(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, posemb, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        embedded = self.embedding(input_var)
        if self.position_embedding is not None:
            if self.pos_add == 'cat':
                embedded = torch.cat((embedded, posemb), dim=2)
            else:
                embedded += posemb
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(
                self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def insert_token(self, vocab):
        insert_tok = list()
        insert_num = [str(i) for i in range(1,110)]
        for k in vocab.stoi.keys():
            if k in insert_num:
                insert_tok.append(vocab.stoi[k])

        return insert_tok

    def sin_encoding(self, inputs,
            batch_size, max_len, inputs_lengths, d_model):
        pe = np.zeros((batch_size, max_len, d_model))
        for batch in range(batch_size):
            for m in range(max_len):
                if inputs_lengths[batch] <= 0:
                    for i in range(0, d_model, 2):
                        pe[batch, m, i] = 0.0
                        if i+1 == d_model:
                            break
                        pe[batch, m, i+1] = 0.0
                else:
                    if inputs[batch][m] not in self.insert_tok:
                        inputs_lengths[batch] -= 1
                    for i in range(0, d_model, 2):
                        pe[batch, m, i] = math.sin((
                            inputs_lengths[batch])/(10000**(i/d_model)))
                        if i+1 == d_model:
                            break
                        pe[batch, m, i+1] = math.cos((
                            inputs_lengths[batch])/(10000**(i/d_model)))
        pos = torch.from_numpy(pe)
        if torch.cuda.is_available():
            pos = pos.type(torch.cuda.FloatTensor)
        return pos, inputs_lengths

    def length_encoding(self, inputs,
            batch_size, max_len, inputs_lengths):
        pe = []
        for batch in range(batch_size):
            p = []
            for i in range(max_len):
                if inputs_lengths[batch] <= 0:
                    p.append(0)
                else:
                    if inputs[batch][i] not in self.insert_tok:
                        inputs_lengths[batch] -= 1
                    p.append(inputs_lengths[batch])
            pe.append(p)
        pos = torch.tensor(pe)
        if torch.cuda.is_available():
            pos = pos.cuda()
        posemb = self.pos_embedding(pos)
        return posemb, inputs_lengths

    def forward(self, inputs=None, input_lengths=None, encoder_hidden=None,
            encoder_outputs=None, function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        ret_dict[DecoderRNN.KEY_ENCODER_OUTPUTS] = encoder_outputs.squeeze(0)
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        # input.shape = batch_size x sequence_length
        # encoder_outputs.shape = batch_size x sequence_length (50) x hidden_size (50 x 2)
        # encoder_hidden = tuple of the last hidden state and the last cell state.
        # Last cell state = number of layers * batch_size * hidden_size
        # Last hidden state = the same as above
        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden,
                                                             encoder_outputs, function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            pos = None
            if self.position_embedding == "sin":
                pos, _ = self.sin_encoding(inputs.cpu().tolist(),
                    batch_size, max_length, input_lengths, self.embedding_size)
            elif self.position_embedding == "length":
                pos, _ = self.length_encoding(inputs.cpu().tolist(),
                    batch_size, max_length, input_lengths)
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn  = self.forward_step(
                    decoder_input, pos, decoder_hidden, encoder_outputs, function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            input_len = copy.deepcopy(input_lengths)
            for di in range(max_length):
                decoder_pos = None
                if self.position_embedding == "sin":
                    pos, input_len = self.sin_encoding(decoder_input.cpu().tolist(),
                        batch_size, 1, input_len, self.embedding_size)
                    decoder_pos = pos[:, 0].unsqueeze(1)
                elif self.position_embedding == "length":
                    pos, input_len = self.length_encoding(decoder_input.cpu().tolist(),
                        batch_size, 1, input_len)
                    decoder_pos = pos[:, 0].unsqueeze(1)

                decoder_output, decoder_hidden, step_attn  = self.forward_step(
                        decoder_input, decoder_pos, decoder_hidden, encoder_outputs, function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
