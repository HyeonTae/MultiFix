from util.helpers import cpp_compilation_errors, make_dir_if_not_exists
import os
import numpy as np
from functools import partial
import json
from tqdm import tqdm
import glob

def generate_training_data(path, validation_users):
    data_path = os.path.join(path, 'err-data-compiler--orig-spoc/')
    result = {'train': {}, 'validation': {}}

    exceptions_in_mutate_call = 0
    check_problem_path = os.path.join(path, 'comp_result')
    make_dir_if_not_exists(check_problem_path)
    new_data_path = os.path.join(path, 'corrupt_data/')
    make_dir_if_not_exists(new_data_path)

    count = 0
    _error = 0

    for data_file in tqdm(glob.glob(data_path+"*")):
        count += 1
        try:
            data = json.loads(open(data_file).read())
        except:
            exceptions_in_mutate_call += 1
            continue

        problem_id = data["meta"]["probid"]
        user_id = data["meta"]["subid"]
        key = 'validation' if problem_id in validation_keys else 'train'

        code_list = []
        code_list.append("#include <bits/stdc++.h>")
        code_list.append("#include <string>")
        code_list.append("using namespace std ;")
        for lines in data["lines"]:
            code_list.append(lines["code"])

        temp_errors, temp_errors_full = cpp_compilation_errors("\n".join(code_list), check_problem_path)

        #print("Code\n{}".format("\n".join(code_list)))
        #print("error num: {}".format(len(temp_errors)))
        #print("Error details\n{}".format(temp_errors_full.decode('utf-8')))

        if len(temp_errors) == 0:
            os.system("cp " + data_file + " " + new_data_path)
        else:
            _error += 1

    print("Exceptions in mutate() call: {}".format(exceptions_in_mutate_call))
    print("Total : {}, Errors : {}".format(count, _error))

if __name__ == '__main__':

    path = os.path.join('data_processing', 'DrRepair_spoc')
    validation_keys = np.load(os.path.join(path, 'validation_keys.npy'))
    generate_training_data(path, validation_keys)
