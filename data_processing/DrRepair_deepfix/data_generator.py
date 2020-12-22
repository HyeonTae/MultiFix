from util.tokenizer import EmptyProgramException
from util.helpers import get_rev_dict, make_dir_if_not_exists
from util.helpers import apply_fix, getTrace, getEditDistance, apply_edits
from util.helpers import tokens_to_source, compilation_errors
from util.c_tokenizer import C_Tokenizer
import os
import time
import argparse
import sqlite3
import numpy as np
from functools import partial
import json
import copy
from tqdm import tqdm
import glob

tokenize = C_Tokenizer().tokenize

with open("data_processing/DrRepair_deepfix/target_vocab.json", "r") as json_file:
    target_vocab = json.load(json_file)

class FixIDNotFoundInSource(Exception):
    pass

def remove_line_numbers(source):
    lines = source.count('~')
    for l in range(lines):
        if l >= 10:
            source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
        else:
            source = source.replace(str(l) + " ~ ", "", 1)
    source = source.replace("  ", " ")
    return source

def get_target(corrupt_program, program):
    log = getTrace(corrupt_program, program, getEditDistance(corrupt_program, program))

    target = ["0" for i in range(len(corrupt_program))]
    for l in log:
        if l[0] == "i":
            target.insert(l[1], target_vocab["insert"][l[2]])
        elif l[0] == "r":
            target[l[1]] = target[l[1]].replace(target[l[1]], target_vocab["replace"][l[2]])
        elif l[0] == "d":
            target[l[1]] = target[l[1]].replace(target[l[1]], "-1")

    return " ".join(target)

def generate_training_data(bins, validation_users):
    data_path = "data_processing/DrRepair_deepfix/err-data-compiler--auto-corrupt--orig-deepfix/"
    result = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    for problem_id in tqdm(problem_list):
        for data_file in glob.glob(data_path+problem_id+"/*"):
            try:
                data = json.loads(open(data_file).read())
            except:
                exceptions_in_mutate_call += 1
                continue

            code_id = data["meta"]["subid"].split("-")[0]
            user_id = data["meta"]["subid"].split("-")[1]
            key = 'validation' if user_id in validation_users[problem_id] else 'train'
            code_list = []
            for lines in data["lines"]:
                code_list.append(lines["code"])
            try:
                tokenized_code, name_dict, name_sequence = tokenize("\n".join(code_list))
            except:
                exceptions_in_mutate_call += 1
                continue
            # Correct pairs
            source = ' '.join(remove_line_numbers(tokenized_code).split())
            target = ["0" for i in range(len(source.split()))]
            try:
                result[key][problem_id] += [
                        (source, name_dict, name_sequence,
                            user_id, code_id, " ".join(target))]
            except:
                result[key][problem_id] = [
                        (source, name_dict, name_sequence,
                            user_id, code_id, " ".join(target))]

            # Mutate
            for iter_i in range(len(data["errors"])):
                temp = copy.deepcopy(code_list)
                for mod_line, mod_code in zip(data["errors"][iter_i]['mod_line'],
                        data["errors"][iter_i]['mod_code']):
                    if len(mod_code) == len(temp[mod_line]):
                        temp[mod_line] = mod_code
                    elif len(mod_code) == len(temp[mod_line])-1:
                        temp[mod_line] = mod_code
                    elif len(mod_code) == len(temp[mod_line])+1:
                        temp[mod_line] = mod_code

                try:
                    corrupt_program, _, _ = tokenize("\n".join(temp), name_dict)
                except:
                    exceptions_in_mutate_call += 1
                    continue
                #source sequence
                corrupt_source = ' '.join(remove_line_numbers(corrupt_program).split())
                #target sequence
                try:
                    target = get_target(corrupt_source.split(), source.split())
                except:
                    exceptions_in_mutate_call += 1
                    continue

                try:
                    result[key][problem_id] += [
                            (corrupt_source, name_dict, name_sequence,
                                user_id, code_id+"_"+str(iter_i), target)]
                except:
                    result[key][problem_id] = [
                            (corrupt_source, name_dict, name_sequence,
                                user_id, code_id+"_"+str(iter_i), target)]

    print("Exceptions in mutate() call: {}".format(exceptions_in_mutate_call))
    return result

if __name__ == '__main__':

    validation_users = np.load(os.path.join('data', 'deepfix_raw_data', 'validation_users.npy')).item()
    bins = np.load(os.path.join('data', 'deepfix_raw_data', 'bins.npy'))

    output_dir = os.path.join('data', 'DrRepair_deepfix')
    make_dir_if_not_exists(os.path.join(output_dir))

    result = generate_training_data(bins, validation_users)

    with open(output_dir+"/data_train.txt", 'w') as train:
        for k in result['train']:
            for i in result['train'][k]:
                train.write("%s\t%s\n" % (i[0], i[5]))
    with open(output_dir+"/data_val.txt", 'w') as val:
        for k in result['validation']:
            for i in result['validation'][k]:
                val.write("%s\t%s\n" % (i[0], i[5]))

    print('\n\n--------------- all outputs written to {} ---------------\n\n'.format(output_directory))
