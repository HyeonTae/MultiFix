import os
import sqlite3
import json
from functools import partial
import numpy as np


def generate_raw_test_data(db_path, bins, min_program_length, max_program_length):
    raw_test_data = {}
    program_lengths = []

    problem_list = []
    for bin_ in bins:
        for problem_id in bin_:
            problem_list.append(problem_id)

    skipped = 0
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT user_id, code_id, tokenized_code, name_dict, name_seq FROM Code " +\
            "WHERE problem_id=? and codelength>? and codelength<? and errorcount>0;"
        for problem_id in problem_list:
            for row in cursor.execute(query, (problem_id, min_program_length, max_program_length)):
                user_id, code_id, tokenized_code = map(str, row[:-2])
                name_dict, name_sequence = json.loads(
                    row[3]), json.loads(row[4])

                program_length = len(tokenized_code.split())
                program_lengths.append(program_length)
                if program_length >= min_program_length and program_length <= max_program_length:
                    try:
                        raw_test_data[problem_id].append(
                            (tokenized_code, name_dict, name_sequence, user_id, code_id))
                    except KeyError:
                        raw_test_data[problem_id] = [
                            (tokenized_code, name_dict, name_sequence, user_id, code_id)]

                else:
                    skipped += 1
                    print('out of length range:', problem_id, user_id, code_id)

    print('problem_count:', len(raw_test_data))
    print('program_count:', sum([len(raw_test_data[problem_id]) for problem_id in raw_test_data]))
    print('discared_problems:', skipped)

    program_lengths = np.sort(program_lengths)

    print('Statistics')
    print('----------')
    print('Program length:  Mean =', np.mean(program_lengths), '\t95th %ile =', \
        program_lengths[int(0.95 * len(program_lengths))])

    return raw_test_data

if __name__ == "__main__":
    max_program_length = 450
    min_program_length = 75

    db_path = os.path.join('data', 'deepfix_raw_data', 'dataset.db')
    bins = np.load(os.path.join('data', 'deepfix_raw_data', 'bins.npy'))

    output_dir = os.path.join('data', 'deepfix_raw_data')

    raw_test_data = generate_raw_test_data(
        db_path, bins, min_program_length, max_program_length)

    # Save raw test dataset
    np.save(os.path.join(output_dir, 'test_raw.npy'), raw_test_data)

    print('test dataset generated!')
