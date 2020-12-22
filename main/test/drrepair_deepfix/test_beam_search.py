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
data_name = "DrRepair_deepfix"
data_type = "raw"
pretrained_dir_name = None
select = "1"
batch_size = 10
iteration = 5

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

beam_size=10
seq2seq = Seq2seq(config, len(src.vocab), tgt.vocab, tgt.sos_id, tgt.eos_id, True, beam_size)

fig_path = "log/test/" + save_path + "_bs"
if not os.path.isdir(fig_path):
    os.mkdir(fig_path)
fig_path = fig_path + "/" + rnn + "_" + str(select)
if not os.path.isdir(fig_path):
    os.mkdir(fig_path)
database_path = fig_path + "/" + data_type
if not os.path.isdir(database_path):
    os.mkdir(database_path)
database = database_path + "/" + data_name + "_" + data_type + ".db"

check_problem_path = fig_path + "/" + data_type
if not os.path.isdir(check_problem_path):
    os.mkdir(check_problem_path)

if torch.cuda.is_available():
    seq2seq.cuda()

for param in seq2seq.parameters():
    param.data.uniform_(-0.08, 0.08)
            
log_path = "log/pth/"+save_path +"_" + rnn + "_" + str(select) + "_model_save.pth"

loaded_model = torch.load(log_path)
for key in loaded_model.keys():
    if key.startswith('decoder'):
        top_k_decoder_key = key[:7] + '.rnn' + key[7:]
        seq2seq.state_dict()[top_k_decoder_key].copy_(loaded_model[key])
    else:
        seq2seq.state_dict()[key].copy_(loaded_model[key])

seq2seq.eval()

predictor = Predictor(seq2seq, input_vocab, output_vocab, output_vocab.stoi[train.fields['tgt'].pad_token])

def get_n_fix_all(program, n=beam_size):
    tgt_seq_list = predictor.predict_n_batch(program, n)

    tgt_final_list = []

    for tgt_seq in tgt_seq_list:
        temp_tgt = []
        inserted = False

        for i in range(n):
            if len(set(tgt_seq[i].split())) > 1:
                temp_tgt.append(tgt_seq[i])
                inserted = True

        if not inserted:
            tgt_final_list.append([tgt_seq[0]])
        else:
            tgt_final_list.append(temp_tgt)

    return tgt_final_list

test_dataset = np.load(os.path.join(
    test_path, 'test_%s.npy' % (data_type))).item()

tonum_data =  sum([len(test_dataset[pid]) for pid in test_dataset]) # Total number of data
print("test_{} data length : {}".format(data_type, tonum_data))

conn = sqlite3.connect(database)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS programs (
                prog_id text NOT NULL,
                user_id text NOT NULL,
                prob_id text NOT NULL,
                code text NOT NULL,
                name_dict text NOT NULL,
                name_seq text NOT NULL,
                PRIMARY KEY(prog_id)
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS iterations (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                fix text NOT NULL,
                PRIMARY KEY(prog_id, iteration)
             )''')

c.execute('''CREATE TABLE IF NOT EXISTS error_messages (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                error_message text NOT NULL,
                FOREIGN KEY(prog_id, iteration, network) REFERENCES iterations(prog_id, iteration, network)
             )''')

sequences_of_programs = {}
fixes_suggested_by_network = {}

def remove_line_numbers(source):
    lines = source.count('~')
    for l in range(lines):
        if l >= 10:
            source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
        else:
            source = source.replace(str(l) + " ~ ", "", 1)
    source = source.replace("  ", " ")
    return source.split()

with open(inverse_vocab_path, "r") as json_file:
    inverse_vocab = json.load(json_file)

with open(target_vocab_path, "r") as json_file:
    target_vocab = json.load(json_file)

def real_edit_length(edits):
    
    inserted = 0
    for i, edit in enumerate(edits):
        if edit != '0' and not is_replace_edit(edit):
            inserted += 1
        
    return len(edits) - inserted

def is_replace_edit(edit):
    return str(edit) in target_vocab['replace'].values()

def apply_edits(source, edits):
    fixed = []
    inserted = 0

    edit_length = real_edit_length(edits)

    for _ in range(len(source) - edit_length):
        edits.append('0')

    for i, edit in enumerate(edits):
        if i - inserted >= len(source):
            break
        if edit == '0':
            fixed.append(source[i - inserted])
        elif edit != '-1':
            fixed.append(inverse_vocab[edit])
            if not is_replace_edit(edit):
                inserted += 1

    return fixed

def check_errors(program, name_dict):
    temp_errors, temp_errors_full = compilation_errors(
            tokens_to_source(program, name_dict, False), check_problem_path)

    return len(temp_errors)

full_repair_id = []
probid = 0
for problem_id, test_programs in tqdm(test_dataset.items()):
    probid += 1
    sequences_of_programs[problem_id] = {}
    fixes_suggested_by_network[problem_id] = {}

    entries = []

    for program, name_dict, name_sequence, user_id, program_id in test_programs:
        sequences_of_programs[problem_id][program_id] = [program]
        fixes_suggested_by_network[problem_id][program_id] = []
        entries.append(
            (program, name_dict, name_sequence, user_id, program_id,))

        c.execute("INSERT OR IGNORE INTO programs VALUES (?,?,?,?,?,?)", (program_id,
                  user_id, problem_id, program, json.dumps(name_dict), json.dumps(name_sequence)))
    totalentr = len(entries)
    for round_ in range(iteration):
        to_delete = []
        input_ = []

        for i, entry in enumerate(entries):
            _, name_dict, _, _, program_id = entry

            if sequences_of_programs[problem_id][program_id][-1] is not None:
                tmp = sequences_of_programs[problem_id][program_id][-1]
                input_.append(remove_line_numbers(tmp))
            else:
                to_delete.append(i)

        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

        assert len(input_) == len(entries)

        if len(input_) == 0:
            break

        cnt = 0
        fixes = []
        for i in range(math.ceil(len(input_)/batch_size)):
            if cnt+batch_size > len(input_):
                fix = get_n_fix_all(input_[cnt:len(input_)])
            else:
                fix = get_n_fix_all(input_[cnt:cnt+batch_size])
            cnt += batch_size
            fixes += fix

        to_delete = []

        # Check Errors & Apply fixes
        for i, entry, q_fix in zip(range(len(fixes)), entries, fixes):
            _, name_dict, _, _, program_id = entry
            init_prog = remove_line_numbers(sequences_of_programs[problem_id][program_id][-1])
            error_temp = check_errors(" ".join(init_prog), name_dict)
            is_fixed = False
            for q, fix in enumerate(q_fix):
                program = " ".join(apply_edits(init_prog, fix.split()))
                if q == 0:
                    first_fix_program = copy.deepcopy(program)
                err_num = check_errors(program, name_dict)
                if err_num == 0:
                    if program_id not in full_repair_id:
                        full_repair_id.append(program_id)
                    c.execute("INSERT OR IGNORE INTO iterations VALUES (?,?,?,?)",
                         (program_id, round_ + 1, data_name, fix))
                    is_fixed = True
                    to_delete.append(i)
                    break
                elif err_num < error_temp:
                    sequences_of_programs[problem_id][program_id].append(program)
                    c.execute("INSERT OR IGNORE INTO iterations VALUES (?,?,?,?)",
                         (program_id, round_ + 1, data_name, fix))
                    is_fixed = True
                    break

            if not is_fixed:
                if sum(list(map(int, q_fix[0].split()))) == 0:
                    to_delete.append(i)
                else:
                    sequences_of_programs[problem_id][program_id].append(first_fix_program)
                    c.execute("INSERT OR IGNORE INTO iterations VALUES (?,?,?,?)",
                        (program_id, round_ + 1, data_name, q_fix[0]))

        to_delete = sorted(to_delete)[::-1]

        for i in to_delete:
            del entries[i]

        sys.stdout.write('\rProbID:%d/%d, Iter:%d/%d, FullRepair status : %d (%.1f%s) \t||\t Total FullRepair status : %d (%.1f%s)'%(
                            probid, len(test_dataset), round_, iteration,
                            len(full_repair_id), len(full_repair_id)/totalentr*100, '%',
                            len(full_repair_id), len(full_repair_id)/tonum_data*100, '%'))

    conn.commit()

conn.commit()
conn.close()


def get_final_results(database):
    with sqlite3.connect(database) as conn:
        c = conn.cursor()

        error_counts = []

        for row in c.execute("SELECT iteration, COUNT(*) FROM error_messages GROUP BY iteration ORDER BY iteration;"):
            error_counts.append(row[1])

        query1 = """SELECT COUNT(*)
        FROM error_messages
        WHERE iteration = 0 AND prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query1):
            initial_errors = row[0]

        query2 = """SELECT COUNT(*)
        FROM error_messages
        WHERE iteration = 10 AND prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query2):
            final_errors = row[0]

        query3 = """SELECT COUNT(DISTINCT prog_id)
        FROM error_message_strings
        WHERE iteration = 10 AND error_message_count = 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        query3_2 = """SELECT DISTINCT prog_id
        FROM error_message_strings
        WHERE iteration = 10 AND error_message_count = 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        for row in c.execute(query3):
            fully_fixed = row[0]

        query4 = """SELECT DISTINCT prog_id, error_message_count FROM error_message_strings
        WHERE iteration = 0 AND error_message_count > 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        query5 = """SELECT DISTINCT prog_id, error_message_count FROM error_message_strings
        WHERE iteration = 10 AND error_message_count > 0 and prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"""

        original_errors = {}
        for row in c.execute(query4):
            original_errors[row[0]] = int(row[1])

        partially_fixed = {}
        unfixed = {}
        for row in c.execute(query5):
            if int(row[1]) < original_errors[row[0]]:
                partially_fixed[row[0]] = int(row[1])
            elif int(row[1]) == original_errors[row[0]]:
                unfixed[row[0]] = int(row[1])
            else:
                print(row[0], row[1], original_errors[row[0]])

        token_counts = []
        assignments = None

        for row in c.execute("SELECT COUNT(DISTINCT prob_id) FROM programs p WHERE prog_id NOT IN (SELECT p.prog_id FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count = 0);"):
            assignments = int(row[0])

        for row in c.execute("SELECT code FROM programs p INNER JOIN error_message_strings e ON p.prog_id = e.prog_id WHERE e.iteration = 0 AND e.error_message_count <> 0;"):
            token_counts += [len(row[0].split())]

        avg_token_count = np.mean(token_counts)

        print("-------")
        print("Assignments: ", assignments)
        print("Program count: ", len(token_counts))
        print("Average token count: ", avg_token_count)
        print("Error messages: ", initial_errors)
        print("-------")
        print("Errors remaining: %d (%.1f" % (final_errors,
              final_errors/initial_errors*100) + "%)")
        print("Reduction in errors: %d (%.1f" % ((initial_errors - final_errors),
              (initial_errors - final_errors)/initial_errors*100) + "%)")
        print("Completely fixed programs: %d (%.1f" % (fully_fixed,
              fully_fixed/len(token_counts)*100) + "%)")
        print("Partially fixed programs: %d (%.1f" % (len(partially_fixed),
              len(partially_fixed)/len(token_counts)*100) + "%)")
        print("Unfixed programs: %d (%.1f" % (len(unfixed),
              len(unfixed)/len(token_counts)*100) + "%)")
        print("-------")

def do_problem(problem_id):
    global reconstruction, errors, errors_full, total_count, errors_test

    c = conn.cursor()

    reconstruction[problem_id] = {}
    errors[problem_id] = {}
    errors_full[problem_id] = {}
    errors_test[problem_id] = []
    candidate_programs = []

    for row in c.execute('SELECT user_id, prog_id, code, name_dict, name_seq FROM programs WHERE prob_id = ?', (problem_id,)):
        user_id, prog_id, initial = row[0], row[1], " ".join(remove_line_numbers(row[2]))
        name_dict = json.loads(row[3])
        name_seq = json.loads(row[4])

        candidate_programs.append(
            (user_id, prog_id, initial, name_dict, name_seq,))

    for _, prog_id, initial, name_dict, name_seq in candidate_programs:
        #fixes_suggested_by_typo_network = []
        #fixes_suggested_by_undeclared_network = []
        fixes_suggested_by_network = []

        #for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? AND network = \'typo\' ORDER BY iteration', (prog_id,)):
        #    fixes_suggested_by_typo_network.append(row[0])

        #for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? AND network = \'ids\' ORDER BY iteration', (prog_id,)):
        #    fixes_suggested_by_undeclared_network.append(row[0])
        
        for row in c.execute('SELECT fix FROM iterations WHERE prog_id=? ORDER BY iteration', (prog_id,)):
            fixes_suggested_by_network.append(row[0])

        reconstruction[problem_id][prog_id] = [initial]
        temp_errors, temp_errors_full = compilation_errors(
            tokens_to_source(initial, name_dict, False), database_path)
        errors[problem_id][prog_id] = [temp_errors]
        errors_full[problem_id][prog_id] = [temp_errors_full]
        
        for fix in fixes_suggested_by_network:
            temp_prog = " ".join(apply_edits(
                reconstruction[problem_id][prog_id][-1].split() , fix.split()))
            temp_errors, temp_errors_full = compilation_errors(
                tokens_to_source(temp_prog, name_dict, False), database_path)
            
            if len(temp_errors) > len(errors[problem_id][prog_id][-1]):
                break
            else:
                reconstruction[problem_id][prog_id].append(temp_prog)
                errors[problem_id][prog_id].append(temp_errors)
                errors_full[problem_id][prog_id].append(
                    temp_errors_full)
        
        while len(reconstruction[problem_id][prog_id]) <= 10:
            reconstruction[problem_id][prog_id].append(
                reconstruction[problem_id][prog_id][-1])
            errors[problem_id][prog_id].append(errors[problem_id][prog_id][-1])
            errors_full[problem_id][prog_id].append(
                errors_full[problem_id][prog_id][-1])

        errors_test[problem_id].append(errors[problem_id][prog_id])

        for k, errors_t, errors_full_t in zip(range(len(errors[problem_id][prog_id])), errors[problem_id][prog_id], errors_full[problem_id][prog_id]):
            c.execute("INSERT INTO error_message_strings VALUES(?, ?, ?, ?, ?)", (
                prog_id, k, 'typo', errors_full_t.decode('utf-8', 'ignore'), len(errors_t)))

            for error_ in errors_t:
                c.execute("INSERT INTO error_messages VALUES(?, ?, ?, ?)",
                            (prog_id, k, 'typo', error_.decode('utf-8', 'ignore'),))

    count_t = len(candidate_programs)
    total_count += count_t
    conn.commit()


    c.close()

conn = sqlite3.connect(database)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS error_message_strings (
                prog_id text NOT NULL,
                iteration text NOT NULL,
                network text NOT NULL,
                error_message_string text NOT NULL,
                error_message_count integer NOT NULL,
                FOREIGN KEY(prog_id, iteration, network) REFERENCES iterations(prog_id, iteration, network)
             )''')

problem_ids = []

for row in c.execute('SELECT DISTINCT prob_id FROM programs'):
    problem_ids.append(row[0])

c.close()

reconstruction = {}
errors = {}
errors_full = {}
errors_test = {}

fixes_per_stage = [0] * 10

total_count = 0

start = time.time()

for problem_id in tqdm(problem_ids):
    do_problem(problem_id)

time_t = time.time() - start

conn.commit()
conn.close()

print('Total time:', time_t, 'seconds')
print('Total programs processed:', total_count)
print('Average time per program:', int(float(time_t) / float(total_count) * 1000), 'ms')

total_fixes_num = {}
errors_before = {}

for problem_id in errors_test:
    total_fixes_num[problem_id] = 0

    for j, seq in enumerate(errors_test[problem_id]):
        error_numbers = [len(x) for x in seq]
        skip = False

        for i in range(len(error_numbers) - 1):
            assert (not error_numbers[i + 1] > error_numbers[i])
            total_fixes_num[problem_id] += error_numbers[i] - \
                error_numbers[i + 1]

            if error_numbers[i] != error_numbers[i + 1]:
                fixes_per_stage[i] += error_numbers[i] - error_numbers[i + 1]

total_numerator = 0
total_denominator = 0

for problem_id in errors_test:
    total_numerator += total_fixes_num[problem_id]
    total_denominator += sum([len(x[0]) for x in errors_test[problem_id]])


print(int(float(total_numerator) * 100.0 / float(total_denominator)), '%')


for stage in range(len(fixes_per_stage)):
    print('Stage', stage, ':', fixes_per_stage[stage])

get_final_results(database)
