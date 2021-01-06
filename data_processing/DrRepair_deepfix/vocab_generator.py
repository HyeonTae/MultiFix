from util.helpers import make_dir_if_not_exists
from util.c_tokenizer import C_Tokenizer
import os
import numpy as np
import json
import copy
from tqdm import tqdm
import glob

tokenize = C_Tokenizer().tokenize

def remove_line_numbers(source):
    lines = source.count('~')
    for l in range(lines):
        if l >= 10:
            source = source.replace(list(str(l))[0] + " " + list(str(l))[1] + " ~ ", "", 1)
        else:
            source = source.replace(str(l) + " ~ ", "", 1)
    source = source.replace("  ", " ")
    return source

def generate_vocab(bins, path, validation_users):
    tok_list = list()
    data_path = os.path.join(path, 'err-data-compiler--auto-corrupt--orig-deepfix/')
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

            user_id = data["meta"]["subid"].split("-")[1]
            key = 'validation' if user_id in validation_users[problem_id] else 'train'

            code_list = []
            origincode = ''
            for lines in data["lines"]:
                origincode += lines["code"] + '\n'
                code_list.append(lines["code"])

            try:
                tokenized_code, name_dict, _ = tokenize("\n".join(code_list))
            except:
                exceptions_in_mutate_call += 1
                continue
            # Correct pairs
            source = ' '.join(remove_line_numbers(tokenized_code).split())
            for a in source.split(' '):
                if a not in tok_list:
                    tok_list.append(a)

            # Mutate
            for iter_i in range(len(data["errors"])):
                temp = copy.deepcopy(code_list)
                for mod_line, mod_code in zip(data["errors"][iter_i]['mod_line'],
                    data["errors"][iter_i]['mod_code']):
                    temp[mod_line] = mod_code

                origincode = "\n".join(temp)
                try:
                    corrupt_program, _, _ = tokenize("\n".join(temp), name_dict)
                except:
                    exceptions_in_mutate_call += 1
                    continue
                #source sequence
                corrupt_source = ' '.join(remove_line_numbers(corrupt_program).split())
                for a in corrupt_source.split(' '):
                    if a not in tok_list:
                        tok_list.append(a)

    tok_list.sort()
    tok_dict = dict()
    tok_dict_re = dict()
    tok_dict["insert"] = dict()
    tok_dict["replace"] = dict()
    for i in range(len(tok_list)):
        tok_dict["insert"][tok_list[i]] = str(i)
        tok_dict_re[str(i)] = tok_list[i]
    for i in range(len(tok_list)):
        tok_dict["replace"][tok_list[i]] = str(i+len(tok_list))
        tok_dict_re[str(i+len(tok_list))] = tok_list[i]

    with open(os.path.join(path, 'target_vocab.json'), 'w') as f:
        json.dump(tok_dict, f, indent=4)
    with open(os.path.join(path, 'target_vocab_reverse.json'), 'w') as f:
        json.dump(tok_dict_re, f, indent=4)
    print("Exceptions in mutate() call: {}".format(exceptions_in_mutate_call))

if __name__ == '__main__':
    path = os.path.join('data_processing', 'DrRepair_deepfix')
    validation_users = np.load(os.path.join('data', 'raw_data', 'validation_users.npy')).item()
    bins = np.load(os.path.join('data', 'raw_data', 'bins.npy'))
    generate_vocab(bins, path, validation_users)
