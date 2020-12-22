#!/usr/bin/env python
# coding: utf-8

# # Set Parameter
# - Attention = True
# - Teacher Forcing Ratio = 0.5
# - Layer = 4
# - Batch size = 128
# - Learning rate = 0.001
# - Hidden unit = 256
# - Epochs = 100

# # Import packages
# import useful packages for experiments
import os
import argparse
import logging
import sys
import json

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))))
#os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))))

from models.trainer import Trainer
from models.seq2seq import Seq2seq
from models.loss import Perplexity
from models import fields

import matplotlib.pyplot as plt


# # Log format
log_level = 'info'
LOG_FORMAT = '%(asctime)s %(levelname)-6s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, log_level.upper()))

character_accuracy = []
sentence_accuracy = []
f1_score = []
rnn = "lstm"
data_name = "DrRepair_deepfix"
iterator = [1]
epochs = 100

train_path = "data/"+data_name+"/data_train.txt"
dev_path = "data/"+data_name+"/data_val.txt"

config_path = "models/config.json"

# # Prepare dataset
for i in iterator:
    print("rnn : %s" % rnn)
    max_len = 400
    src = fields.SourceField()
    srcp = fields.SourceField()
    tgt = fields.TargetField()
    tgtp = fields.TargetField()
    def len_filter(example):
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    train = torchtext.data.TabularDataset(
        path=train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    src.build_vocab(train)
    tgt.build_vocab(train)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    print("src vocab size = %d" % (len(src.vocab)))
    print("tat vacab size = %d" % (len(tgt.vocab)))

    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    loss = Perplexity(weight, pad)
    if torch.cuda.is_available():
        loss.cuda()

    optimizer = "Adam"
    seq2seq = None
    config_json = open(config_path).read()
    config = json.loads(config_json)
    print(json.dumps(config, indent=4))

    save_path = (data_name
                    + ("_att" if config["use_attention"] else "")
                    + ("_with_pos_" + config["position_embedding"]
                        if config["position_embedding"] is not None else "")
                    + ("_cat" if config["pos_add"] == "cat" else "")
                    + "_emb" + str(config["embedding_size"])
                    + "_hidden" + str(config["hidden_size"])
                    + ("_bi" if config["bidirectional"] else "")
                    + ("_" + str(config["n_layers"]) if config["n_layers"] != 1 else ""))
    print("Save_path : %s" % save_path)
    
    seq2seq = Seq2seq(config, len(src.vocab), tgt.vocab, tgt.sos_id, tgt.eos_id)
    
    if torch.cuda.is_available():
        seq2seq.cuda()

    for param in seq2seq.parameters():
        param.data.uniform_(-0.08, 0.08)

    # train
    t = Trainer(loss=loss, batch_size=128,
                learning_rate=0.001,
                checkpoint_every=50,
                print_every=100,
                hidden_size=config["hidden_size"],
                path=save_path,
                file_name=config["rnn_cell"] + "_" + str(i))

    seq2seq, ave_loss, character_accuracy_list, sentence_accuracy_list, f1_score_list = t.train(seq2seq, train,
                                                                             num_epochs=epochs, dev_data=dev,
                                                                             optimizer=optimizer,
                                                                             teacher_forcing_ratio=0.5)

    character_accuracy.append(character_accuracy_list)
    sentence_accuracy.append(sentence_accuracy_list)
    f1_score.append(f1_score_list)
        
# svae plot
#------- Sentence Accuracy -------
plt.figure(figsize=(15,7))
for j in range(len(sentence_accuracy)):
    plt.plot(list(range(1, len(sentence_accuracy[j])+1, 1))[::3], sentence_accuracy[j][::3], '-', LineWidth=3, label=str(j+1))

plt.legend(loc="best", fontsize=12)
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('Sentence Accuracy', fontsize=24)
plt.ylim([0, 1.02])
plt.grid()
plt.savefig("log/plot/"+save_path+"/"+rnn+"_sentence_accuracy.png")

#------- F1 Score -------
plt.figure(figsize=(15,7))
for j in range(len(f1_score)):
    plt.plot(list(range(1, len(f1_score[j])+1, 1))[::3], f1_score[j][::3], '-', LineWidth=3, label=str(j+1))

plt.legend(loc="best", fontsize=12)
plt.xlabel('Epoch', fontsize=24)
plt.ylabel('F1 Score', fontsize=24)
plt.ylim([0, 1.02])
plt.grid()
plt.savefig("log/plot/"+save_path+"/"+rnn+"_f1_score.png")

print("\n\nFinish!")
