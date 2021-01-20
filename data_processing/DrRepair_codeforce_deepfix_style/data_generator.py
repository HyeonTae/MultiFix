from util.helpers import apply_fix, getTrace, getEditDistance, apply_edits
from util.helpers import cpp_compilation_errors, make_dir_if_not_exists, tokens_to_source
from util.c_tokenizer import C_Tokenizer
import os
import time
import argparse
import sqlite3
import numpy as np
from functools import partial
import json
import random
import copy
from tqdm import tqdm
import glob

tokenize = C_Tokenizer().tokenize

with open("data_processing/target_vocab.json", "r") as json_file:
    target_vocab = json.load(json_file)

def remove_line_numbers(source):
    lines = source.count('~')
    for l in range(lines):
        if l >= 100:
            source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " " + list(str(l))[2] + " ~ ", "", 1)
        elif l >= 10:
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

def generate_training_data(path, validation_users):
    data_path = os.path.join(path, 'err-data-compiler--auto-corrupt--codeforce--deepfix-style/')

    exceptions_in_mutate_call = list()
    check_problem_path = os.path.join(path, 'comp_result')
    make_dir_if_not_exists(check_problem_path)

    count = 0
    _error = list()
    _token_error = list()
    large_token = 0
    result = {'train': {}, 'validation': {}}

    for dir in tqdm(glob.glob(data_path+"*")):
        for data_file in glob.glob(dir+"/*"):
            count += 1
            try:
                data = json.loads(open(data_file).read())
            except:
                exceptions_in_mutate_call.append(data_file)
                continue

            problem_id = data["meta"]["probid"]
            user_id = data["meta"]["subid"]
            key = 'validation' if problem_id in validation_keys else 'train'

            code_list = dict()
            for lines in data["lines"]:
                code_list[lines["line"]] = lines["code"]

            try:
                tokenized_code, name_dict, name_sequence, num_dict, num_seq = tokenize("\n".join(code_list.values()))
            except:
                exceptions_in_mutate_call.append(data_file)
                continue

            # compilation error check
            orig_code = tokens_to_source(tokenized_code, name_dict, num_dict, False)
            orig_code = orig_code.replace(';', ';\n')
            orig_code = orig_code.replace('{', '{\n')
            orig_code = orig_code.replace('}', '}\n')
            orig_code_list = orig_code.split('\n')
            temp_errors, _ = cpp_compilation_errors("\n".join(orig_code_list), check_problem_path)
            if len(temp_errors) != 0:
                _token_error.append(data_file)
                continue

            # Correct pairs
            source = ' '.join(remove_line_numbers(tokenized_code).split())
            if len(source.split()) > 400:
                large_token += 1
                continue
            target = ["0" for i in range(len(source.split()))]
            try:
                result[key][problem_id] += [
                    (source, name_dict, name_sequence,
                        user_id, " ".join(target))]
            except:
                result[key][problem_id] = [
                    (source, name_dict, name_sequence,
                        user_id, " ".join(target))]

            # Mutate
            if len(data["errors"]) > 5:
                ite = random.sample(range(len(data["errors"])), 5)
            else:
                ite = random.sample(range(len(data["errors"])), len(data["errors"]))

            for iter_i in ite:
                temp = copy.deepcopy(code_list)
                for mod_line, mod_code in zip(data["errors"][iter_i]['mod_line'],
                    data["errors"][iter_i]['mod_code']):
                    temp[mod_line] = mod_code
                try:
                    corrupt_program, corrupt_name_dict, _, corrupt_num_dict, _ = tokenize("\n".join(temp.values()), name_dict, num_dict)
                except:
                    exceptions_in_mutate_call.append(data_file)
                    continue

                #source sequence
                corrupt_source = ' '.join(remove_line_numbers(corrupt_program).split())
                if len(corrupt_source.split()) > 400:
                    large_token += 1
                    continue

                #target sequence
                try:
                    target = get_target(corrupt_source.split(), source.split())
                except:
                    exceptions_in_mutate_call.append(data_file)
                    continue

                try:
                    result[key][problem_id] += [
                            (corrupt_source, corrupt_name_dict, name_sequence,
                                user_id+"_"+str(iter_i), target)]
                except:
                    result[key][problem_id] = [
                            (corrupt_source, corrupt_name_dict, name_sequence,
                                user_id+"_"+str(iter_i), target)]

    print("Exceptions in mutate() call: {}".format(len(exceptions_in_mutate_call)))
    print("\n".join(exceptions_in_mutate_call))
    print("-----------------------------------------------------")
    print("Total data: {}".format(count))
    print("-----------------------------------------------------")
    print("Error data: {}".format(len(_error)))
    print("\n".join(_error))
    print("-----------------------------------------------------")
    print("Token Error data: {}".format(len(_token_error)))
    print("\n".join(_token_error))
    print("-----------------------------------------------------")
    print("Data with too large token size: {}".format(large_token))
    return result

if __name__ == '__main__':
    path = os.path.join('data_processing', 'DrRepair_codeforce_deepfix_style')
    validation_keys = np.load(os.path.join(path, 'validation_keys.npy'))

    output_dir = os.path.join('data', 'DrRepair_codeforce_deepfix_style')
    make_dir_if_not_exists(os.path.join(output_dir))

    result = generate_training_data(path, validation_keys)

    with open(output_dir+"/data_train.txt", 'w') as train:
        for k in result['train']:
            for i in result['train'][k]:
                train.write("%s\t%s\n" % (i[0], i[4]))
    with open(output_dir+"/data_val.txt", 'w') as val:
        for k in result['validation']:
            for i in result['validation'][k]:
                val.write("%s\t%s\n" % (i[0], i[4]))

    print('\n\n--------------- all outputs written to {} ---------------\n\n'.format(output_dir))
