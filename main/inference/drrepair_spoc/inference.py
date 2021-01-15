import os
import argparse
import logging
import sys
import numpy as np
import json
import math
import time
import sqlite3
import regex as re
from tqdm import tqdm
import copy

#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))))
os.chdir(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))))))
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

from models.seq2seq import Seq2seq
from models.loss import Perplexity
from models.optim import Optimizer
from models import fields
from evaluator.predictor import Predictor
from util.helpers import tokens_to_source, compilation_errors

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, "info".upper()))

rnn = "lstm"
data_name = "DrRepair_spoc"
data_type = "raw"
pretrained_dir_name = None
select = "1"
batch_size = 64

train_path = "data/"+data_name+"/data_train.txt"
test_path = "data/deepfix_raw_data"
config_path = "models/config.json"

target_vocab_path = "data_processing/"+data_name+"/target_vocab.json"
inverse_vocab_path = "data_processing/"+data_name+"/target_vocab_reverse.json"

max_len = 400
src = fields.SourceField()
tgt = fields.TargetField()
def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len
train = torchtext.data.TabularDataset(
    path=train_path, format='tsv',
    fields=[('src', src), ('tgt', tgt)],
    filter_pred=len_filter)
src.build_vocab(train)
tgt.build_vocab(train)
input_vocab = src.vocab
output_vocab = tgt.vocab

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

log_path = "log/pth/"+save_path +"_" + rnn + "_" + str(select) + "_model_save.pth"

seq2seq.load_state_dict(torch.load(log_path))
seq2seq.eval()

predictor = Predictor(seq2seq, input_vocab, output_vocab, output_vocab.stoi[train.fields['tgt'].pad_token])

def get_fix(program):
    tgt_seq = predictor.predict_batch(program)
    return tgt_seq

with open(inverse_vocab_path, "r") as json_file:
    inverse_vocab = json.load(json_file)

with open(target_vocab_path, "r") as json_file:
    target_vocab = json.load(json_file)

def is_replace_edit(edit):
    return str(edit) in target_vocab['replace'].values()

def apply_edits(source, edits):
    fixed = []
    inserted = 0

    for i, edit in enumerate(edits):
        if i - inserted >= len(source):
            break
        if edit == '0':
            fixed.append(source[i - inserted])
        elif edit != '-1':
            fixed.append(inverse_vocab[edit])
            if not is_replace_edit(edits[i]):
                inserted += 1
    return fixed

import random

with open("data/"+data_name+"/data_val.txt", 'r') as f:
    data = f.read()
val = data.split('\n')[:-1]
rad_val = random.choice(val)

prediction_result = get_fix([rad_val.split('\t')[0].split()])

corrupt_name_dict = dict()
for i in rad_val.split('\t')[0].split():
    if '_<id>_' in i and i not in corrupt_name_dict.keys():
        corrupt_name_dict[i] = "ID_" + i.split('_')[2].split('@')[0]
corrupt_code = tokens_to_source(rad_val.split('\t')[0], corrupt_name_dict, False)
corrupt_code = corrupt_code.replace(';', ';\n')
corrupt_code = corrupt_code.replace('{', '{\n')
corrupt_code = corrupt_code.replace('}', '}\n')

orig_data = apply_edits(rad_val.split('\t')[0].split(), rad_val.split('\t')[1].split())
orig_name_dict = dict()
for i in orig_data:
    if '_<id>_' in i and i not in orig_name_dict.keys():
        orig_name_dict[i] = "ID_" + i.split('_')[2].split('@')[0]
orig_code = tokens_to_source(' '.join(orig_data), orig_name_dict, False)
orig_code = orig_code.replace(';', ';\n')
orig_code = orig_code.replace('{', '{\n')
orig_code = orig_code.replace('}', '}\n')

print("[Corrupt Data]\n{}".format(rad_val.split('\t')[0]))
print("------------------------------------------------------------------------------------\n")
print("[Corrupt Code]\n{}".format(corrupt_code))
print("------------------------------------------------------------------------------------\n")
print("[Origin Code]\n{}".format(orig_code))
print("------------------------------------------------------------------------------------\n")
print("[Corrupt Target]\n{}".format(rad_val.split('\t')[1]))
print("------------------------------------------------------------------------------------\n")
print("[Predict Result]\n{}".format(prediction_result[0]))
print("------------------------------------------------------------------------------------\n")
