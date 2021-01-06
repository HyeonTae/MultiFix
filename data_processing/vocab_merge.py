import json
import argparse

par = argparse.ArgumentParser()
par.add_argument("-t", "--types", default="deepfix", choices=["deepfix", "spoc"],
                 type=str, help="Choose a data type. (deepfix/spoc)")
args = par.parse_args()

data_1 = 'DrRepair_' + args.types
data_2 = 'DrRepair_codeforce_' + args.types + '_style'

with open(data_1 + '/orig_target_vocab.json', 'r') as f:
    vocab_1 = json.load(f)

with open(data_2 + '/orig_target_vocab.json', 'r') as f:
    vocab_2 = json.load(f)

vocab_1_i = vocab_1["insert"]
vocab_2_i = vocab_2["insert"]

merged = {**vocab_1["insert"], **vocab_2["insert"]}
tok_list = sorted(merged.keys())

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

with open(data_1 + '/target_vocab.json', 'w') as f:
    json.dump(tok_dict, f, indent=4)

with open(data_1 + '/target_vocab_reverse.json', 'w') as f:
    json.dump(tok_dict_re, f, indent=4)

with open(data_2 + '/target_vocab.json', 'w') as f:
    json.dump(tok_dict, f, indent=4)

with open(data_2 + '/target_vocab_reverse.json', 'w') as f:
    json.dump(tok_dict_re, f, indent=4)
