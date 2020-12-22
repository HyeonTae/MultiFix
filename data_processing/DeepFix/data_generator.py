from util.tokenizer import EmptyProgramException
from util.helpers import get_rev_dict, make_dir_if_not_exists
from util.helpers import apply_fix, getTrace, getEditDistance, apply_edits
from util.helpers import tokens_to_source, compilation_errors
import os
import time
import argparse
import sqlite3
import numpy as np
from functools import partial
import json
from tqdm import tqdm

with open("data_processing/DeepFix/target_vocab.json", "r") as json_file:
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

def get_typo_target(corrupt_program, program):
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

def get_ids_target(program, fix, target):
    source = ' '.join(remove_line_numbers(program).split(' '))
    inset_line = int(fix.split(' ~ ')[0].split()[1])+1
    if inset_line >= 10:
        insert_position = list(str(inset_line))[0] + " " + list(str(inset_line))[1] + " ~"
    else:
        insert_position = str(inset_line) + " ~"
    fixed = program.replace(insert_position, 'insert ' + insert_position, 1)
    fixed = ' '.join(remove_line_numbers(fixed).split(' '))
    fixed_list = fixed.split()
    insert_start_index = fixed_list.index('insert')
    insert_string = fix.split(" ~ ")[1].split()
    for index in range(len(insert_string)):
        target.insert(insert_start_index + index, target_vocab["insert"][insert_string[index]])

    return target

def rename_ids_(rng, corrupt_program, fixes):
    corrupt_program_new = ''
    fixes_new = []

    names = []
    for token in corrupt_program.split():
        if '_<id>_' in token:
            if token not in names:
                names.append(token)

    rng.shuffle(names)
    name_dictionary = {}

    for token in corrupt_program.split():
        if '_<id>_' in token:
            if token not in name_dictionary:
                name_dictionary[token] = '_<id>_' + \
                    str(names.index(token) + 1) + '@'

    for fix in fixes:
        for token in fix.split():
            if '_<id>_' in token:
                if token not in name_dictionary:
                    raise FixIDNotFoundInSource

    # Rename
    for token in corrupt_program.split():
        if '_<id>_' in token:
            corrupt_program_new += name_dictionary[token] + " "
        else:
            corrupt_program_new += token + " "

    for fix in fixes:
        fix_new = ''
        for token in fix.split():
            if '_<id>_' in token:
                fix_new += name_dictionary[token] + " "
            else:
                fix_new += token + " "
        fixes_new.append(fix_new)
    return corrupt_program_new, fixes_new

def generate_training_data(db_path, bins, validation_users, min_program_length, max_program_length,
                           max_fix_length, kind_mutations, max_mutations, max_variants, seed):
    rng = np.random.RandomState(seed)

    if kind_mutations == 'typo':
        from data_processing.DeepFix_style.typo_mutator import LoopCountThresholdExceededException, FailedToMutateException, Typo_Mutate, typo_mutate
        mutator_obj = Typo_Mutate(rng)
        mutate = partial(typo_mutate, mutator_obj)
        op = "replace"

        def rename_ids(x, y): return (x, y)
    else:
        from data_processing.DeepFix_style.undeclared_mutator import LoopCountThresholdExceededException, FailedToMutateException, id_mutate
        mutate = partial(id_mutate, rng)
        rename_ids = partial(rename_ids_, rng)
        op = "insert"

    result = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0
    total_mutate_calls = 0
    program_lengths, fix_lengths = [], []

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT user_id, code_id, tokenized_code, name_dict, name_seq FROM Code " +\
            "WHERE problem_id=? and codelength>? and codelength<? and errorcount=0;"
        for problem_id in tqdm(problem_list):
            for row in cursor.execute(query, (problem_id, min_program_length, max_program_length)):
                user_id, code_id, tokenized_code = map(str, row[:-2])
                name_dict, name_sequence = json.loads(
                    row[3]), json.loads(row[4])
                key = 'validation' if user_id in validation_users[problem_id] else 'train'

                program_length = len(tokenized_code.split())
                program_lengths.append(program_length)

                if program_length >= min_program_length and program_length <= max_program_length:
                    id_renamed_correct_program, _ = rename_ids(
                        tokenized_code, '')

                    # Correct pairs
                    dummy_fix_for_correct_program = '-1'
                    source = ' '.join(remove_line_numbers(id_renamed_correct_program).split(' ')[:-1])
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
                    total_mutate_calls += 1
                    try:
                        iterator = mutate(
                            tokenized_code, max_mutations, max_variants)

                    except FailedToMutateException:
                        exceptions_in_mutate_call += 1
                    except LoopCountThresholdExceededException:
                        exceptions_in_mutate_call += 1
                    except ValueError:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    except AssertionError:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    except Exception:
                        exceptions_in_mutate_call += 1
                        if kind_mutations == 'typo':
                            raise
                    else:
                        for iter_i, temp in enumerate(iterator):
                            if kind_mutations == 'typo':
                                temp = [temp]
                                
                            for temp_i, (corrupt_program, fixes) in enumerate(temp):
                                corrupt_program_length = len(
                                    corrupt_program.split())
                                fix_length = len(fixes[-1].split())
                                fix_lengths.append(fix_length)

                                if corrupt_program_length >= min_program_length and \
                                   corrupt_program_length <= max_program_length and fix_length <= max_fix_length:

                                    try:
                                        corrupt_program, fixes = rename_ids(
                                            corrupt_program, fixes)
                                    except FixIDNotFoundInSource:
                                        exceptions_in_mutate_call += 1

                                    #source sequence
                                    corrupt_source = ' '.join(remove_line_numbers(
                                        corrupt_program).split(' ')[:-1])
                                    target = list()
                                    for l in range(len(corrupt_source.split())):
                                        target.append("0")
                                    #target sequence
                                    if kind_mutations != 'typo':
                                        for fix in fixes:
                                            target = get_ids_target(corrupt_program, fix, target)
                                            corrupt_program = apply_fix(corrupt_program, fix, "insert")
                                        target = " ".join(target)

                                    else:
                                        target = get_typo_target(corrupt_source.split(), source.split())

                                    try:
                                        result[key][problem_id] += [
                                            (corrupt_source, name_dict, name_sequence,
                                                user_id, code_id+"_"+str(iter_i)+"_"+str(temp_i), target)]
                                    except:
                                        result[key][problem_id] = [
                                            (corrupt_source, name_dict, name_sequence,
                                                user_id, code_id+"_"+str(iter_i)+"_"+str(temp_i), target)]

    program_lengths = np.sort(program_lengths)
    fix_lengths = np.sort(fix_lengths)

    print('Statistics')
    print('----------')
    print('Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', program_lengths[int(0.95 * len(program_lengths))])
    try:
        print('Mean fix length: Mean =', np.mean(fix_lengths), '\t95th %ile = ', fix_lengths[int(0.95 * len(fix_lengths))])
    except Exception as e:
        print(e)
        print('fix_lengths')
        print(fix_lengths)
    print('Total mutate calls:', total_mutate_calls)
    print('Exceptions in mutate() call:', exceptions_in_mutate_call, '\n')

    return result

if __name__ == '__main__':
    max_program_length = 450
    min_program_length = 75
    max_fix_length = 25
    max_mutations = 5

    db_path = os.path.join('data', 'deepfix_raw_data', 'dataset.db')
    validation_users = np.load(os.path.join('data', 'deepfix_raw_data', 'validation_users.npy')).item()
    bins = np.load(os.path.join('data', 'deepfix_raw_data', 'bins.npy'))

    seed = 1189

    output_directory = os.path.join('data', 'DeepFix')
    make_dir_if_not_exists(os.path.join(output_directory))
    for kind_mutations in ['typo', 'ids']:
        max_variants = 4 if kind_mutations == 'ids' else 2
        output_dir = os.path.join(output_directory, kind_mutations)
        make_dir_if_not_exists(os.path.join(output_dir))

        result = generate_training_data(db_path, bins, validation_users,
                                    min_program_length, max_program_length, max_fix_length,
                                    kind_mutations, max_mutations, max_variants, seed)

        with open(output_dir+"/data_train.txt", 'w') as train:
            for k in result['train']:
                for i in result['train'][k]:
                    train.write("%s\t%s\n" % (i[0], i[5]))
        with open(output_dir+"/data_val.txt", 'w') as val:
            for k in result['validation']:
                for i in result['validation'][k]:
                    val.write("%s\t%s\n" % (i[0], i[5]))

        print('\n\n--------------- outputs written to {} ---------------\n\n'.format(output_directory))
