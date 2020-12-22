"""
Copyright 2017 Rahul Gupta, Soham Pal, Aditya Kanade, Shirish Shevade.
Indian Institute of Science.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from util.helpers import fix_ids_are_in_program, isolate_line, extract_line_number, tokens_to_source, \
    get_rev_dict, get_lines, recompose_program
import numpy as np
import regex as re


class FailedToMutateException(Exception):
    pass


class CouldNotFindUsesForEitherException(Exception):
    pass


class NothingToMutateException(FailedToMutateException):
    pass


class LoopCountThresholdExceededException(Exception):
    pass


def which_fix_goes_first(program, fix1, fix2):
    try:
        fix1_location = extract_line_number(' '.join(fix1.split()[1:]))
        fix2_location = extract_line_number(' '.join(fix2.split()[1:]))
    except Exception:
        raise

    if not fix_ids_are_in_program(recompose_program(get_lines(program)[fix2_location:]), fix2) and fix_ids_are_in_program(recompose_program(get_lines(program)[fix1_location:]), fix1):
        return fix1

    if not fix_ids_are_in_program(recompose_program(get_lines(program)[fix1_location:]), fix1) and fix_ids_are_in_program(recompose_program(get_lines(program)[fix2_location:]), fix2):
        return fix2

    if not fix_ids_are_in_program(recompose_program(get_lines(program)[fix1_location:]), fix1) and not fix_ids_are_in_program(recompose_program(get_lines(program)[fix2_location:]), fix2):
        raise CouldNotFindUsesForEitherException

    if fix1_location < fix2_location:
        return fix1
    elif fix2_location < fix1_location:
        return fix2

    prog_lines = get_lines(program)
    id_in_fix1 = None
    id_in_fix2 = None

    for token in fix1.split():
        if '_<id>_' in token:
            assert id_in_fix1 is None, fix1
            id_in_fix1 = token
        elif token == '_<op>_[':
            break

    for token in fix2.split():
        if '_<id>_' in token:
            assert id_in_fix2 is None, fix2
            id_in_fix2 = token
        elif token == '_<op>_[':
            break

    assert id_in_fix1 != id_in_fix2, fix1 + ' & ' + fix2
    assert fix1_location == fix2_location

    for i in range(fix1_location, len(prog_lines)):
        for token in prog_lines[i].split():
            if token == id_in_fix1:
                return fix1
            elif token == id_in_fix2:
                return fix2

    assert False, 'unreachable code'
    raise CouldNotFindUsesForEitherException


def undeclare_variable(rng, old_program, program_string, deleted_ids, name_dict=None, print_debug_messages=False):
    if name_dict is not None:
        rev_name_dict = get_rev_dict(name_dict)

    # Lines
    orig_lines = get_lines(program_string)
    old_lines = get_lines(old_program)

    # Lines to ignore
    struct_lines = []
    structs_deep = 0

    for i, line in enumerate(orig_lines):
        if len(re.findall('_<keyword>_struct _<id>_\d@ _<op>_\{', line)) > 0 or \
           len(re.findall('_<keyword>_union _<id>_\d@ _<op>_\{', line)) > 0 or \
           len(re.findall('_<keyword>_enum _<id>_\d@ _<op>_\{', line)) > 0:
            structs_deep += len(re.findall('_<op>_\{', line))
        elif structs_deep > 0:
            structs_deep += len(re.findall('_<op>_\{', line))
            structs_deep -= len(re.findall('_<op>_\}', line))
            assert structs_deep >= 0, str(structs_deep) + " " + line
            struct_lines.append(i)

    global_lines = []
    brackets_deep = 0

    for i, line in enumerate(orig_lines):
        if len(re.findall('_<op>_\{', line)) > 0 or len(re.findall('_<op>_\}', line)) > 0:
            brackets_deep += len(re.findall('_<op>_\{', line))
            brackets_deep -= len(re.findall('_<op>_\}', line))
            assert brackets_deep >= 0, str(brackets_deep) + " " + line
        elif brackets_deep == 0:
            global_lines.append(i)

    if print_debug_messages:
        print('Ignoring lines:', struct_lines)
        print('Ignoring lines:', global_lines)

        for line in sorted(set(struct_lines + global_lines)):
            print("-", orig_lines[line])

    # Variables
    variables = []

    for token in program_string.split():
        if '_<id>_' in token:
            if token not in variables:
                variables.append(token)

    assert len(orig_lines) == len(old_lines)

    # Look for a declaration
    done = False

    rng.shuffle(variables)

    for to_undeclare in variables:
        if print_debug_messages:
            print('Looking for:', rev_name_dict[to_undeclare], '...')

        # Find a location (scope) to undeclare it from
        shuffled_lines = list(set(range(len(orig_lines))) -
                              set(struct_lines + global_lines))
        rng.shuffle(shuffled_lines)

        # NEW
        regex_alone_use = '(_<keyword>_(?:struct|enum|union) _<id>_\d+@|_<type>_\w+)((?: _<op>_\*)* %s(?: _<op>_\[(?: [^\]]+)? _<op>_\])*)(?: _<op>_= [^,;]+)(?: _<op>_;)' % to_undeclare
        regex_alone = '((?:_<keyword>_(?:struct|enum|union) _<id>_\d+@|_<type>_\w+)(?: _<op>_\*)* %s(?: _<op>_\[(?: [^\]]+)? _<op>_\])* _<op>_;)' % to_undeclare
        regex_group_leader = '((?:_<keyword>_(?:struct|enum|union) _<id>_\d+@|_<type>_\w+)(?: _<op>_\*)*)( %s(?: _<op>_\[(?: [^\]]+)? _<op>_\])*)(?: _<op>_= [^,;]+)?( _<op>_,)(?:(?: _<op>_\*)* _<id>_\d+@(?: _<op>_\[(?: [^\]]+)? _<op>_\])*(?: _<op>_= [^,;]+)? _<op>_,)*(?:(?: _<op>_\*)* _<id>_\d+@(?: _<op>_\[(?: [^\]]+)? _<op>_\])*(?: _<op>_= [^,;]+)? _<op>_;)' % to_undeclare
        regex_group = '(_<keyword>_(?:struct|enum|union) _<id>_\d+@|_<type>_\w+)(?: _<op>_\*)* _<id>_\d+@(?: _<op>_\[(?: [^\]]+)? _<op>_\])*(?: _<op>_= [^,;]+)?(?: _<op>_,(?: _<op>_\*)* _<id>_\d+@(?: _<op>_\[(?: [^\]]+)? _<op>_\])*(?: _<op>_= [^,;]+)?)*( _<op>_,(?: _<op>_\*)* %s(?: _<op>_\[(?: [^\]]+)? _<op>_\])*(?: _<op>_= [^,;]+)?)(?: _<op>_,(?: _<op>_\*)* _<id>_\d+@(?: _<op>_\[(?: [^\]]+)? _<op>_\])*(?: _<op>_= [^,;]+)?)*(?: _<op>_;)' % to_undeclare

        fix_line = None
        declaration = None
        declaration_pos = None

        # Start our search upwards
        for i in shuffled_lines:
            if len(re.findall(regex_alone_use, orig_lines[i])) == 1:
                if print_debug_messages:
                    print(("On line %d:" % i), tokens_to_source(
                        orig_lines[i], name_dict, clang_format=True))
                    print("Found Alone use", re.findall(regex_alone_use, orig_lines[i]))
                m = re.search(regex_alone_use, orig_lines[i])
                declaration = orig_lines[i][m.start(1):m.end(2)] + ' _<op>_;'
                declaration_pos = i

                # Mutate
                orig_lines[i] = orig_lines[i][:m.start(
                    1)] + orig_lines[i][m.end(1) + 1:]
                done = True
                break

            if len(re.findall(regex_alone, orig_lines[i])) == 1:
                if print_debug_messages:
                    print(("On line %d:" % i), tokens_to_source(
                        orig_lines[i], name_dict, clang_format=True))
                    print("Found Alone", re.findall(regex_alone, orig_lines[i]))
                m = re.search(regex_alone, orig_lines[i])
                declaration = orig_lines[i][m.start(1):m.end(1)]
                declaration_pos = i

                # Mutate
                orig_lines[i] = orig_lines[i][:m.start(
                    1)] + orig_lines[i][m.end(1) + 1:]
                done = True
                break

            elif len(re.findall(regex_group, orig_lines[i])) == 1:
                if print_debug_messages:
                    print(("On line %d:" % i), tokens_to_source(
                        orig_lines[i], name_dict, clang_format=True))
                    print("Found Group", re.findall(regex_group, orig_lines[i]))
                m = re.search(regex_group, orig_lines[i])
                declaration = orig_lines[i][m.start(1):m.end(
                    1)] + orig_lines[i][m.start(2):m.end(2)][8:] + ' _<op>_;'
                declaration_pos = i

                try:
                    end_of_declr = declaration.index('_<op>_=')
                    declaration = declaration[:end_of_declr]
                except ValueError:
                    pass

                # Mutate
                orig_lines[i] = orig_lines[i][:m.start(
                    2) + 1] + orig_lines[i][m.end(2) + 1:]
                done = True
                break

            elif len(re.findall(regex_group_leader, orig_lines[i])) == 1:
                if print_debug_messages:
                    print(("On line %d:" % i), tokens_to_source(
                        orig_lines[i], name_dict, clang_format=True))
                    print("Found Group Leader", re.findall(regex_group_leader, orig_lines[i]))
                m = re.search(regex_group_leader, orig_lines[i])
                declaration = orig_lines[i][m.start(1):m.end(2)] + ' _<op>_;'
                declaration_pos = i

                # Mutate
                orig_lines[i] = orig_lines[i][:m.start(
                    2) + 1] + orig_lines[i][m.end(3) + 1:]
                done = True
                break

        if done:
            break

    if not done:
        # Failed to find something to undeclare
        raise NothingToMutateException

    # Find the function signature
    fn_regex = '(?:_<keyword>_(?:struct|union|enum) _<id>_\d+@|_<type>_\w+|_<keyword>_void)(?: _<op>_\*)* (?:_<id>_\d+@|_<APIcall>_main) _<op>_\('
    fn_start_regex = '_<op>_\{'
    inserted = False

    assert declaration_pos != None
    for i in range(declaration_pos, 0, -1):
        if len(re.findall(fn_regex, old_lines[i])) == 1:
            for j in range(i, len(old_lines)):
                if len(re.findall(fn_start_regex, old_lines[i])) >= 1:
                    fix_line = j
                    break
            inserted = True

        if inserted:
            break

    if not inserted:
        # print Failed to insert fix
        raise FailedToMutateException
    if fix_line is None:
        # Couldn't find { after function definition
        raise FailedToMutateException

    fix = '_<insertion>_ '

    assert fix_line is not None

    for digit in str(fix_line):
        fix += str(digit) + ' '

    fix += '~ ' + declaration

    to_delete = False

    if orig_lines[declaration_pos].strip() == '':
        to_delete = declaration_pos
        del orig_lines[to_delete]

    recomposed_program = ''

    for i, line in enumerate(orig_lines):
        for digit in str(i):
            recomposed_program += digit + ' '

        recomposed_program += '~ '
        recomposed_program += line + ' '

    return recomposed_program, fix, fix_line


def get_min(alist):
    if len(alist) == 0:
        return None, None
    m, mi = alist[0], 0
    for idx in range(len(alist)):
        if alist[idx] < m:
            m = alist[idx]
            mi = idx
        pass
    return m, mi


pass


def find_and_replace(rng, org_prog, corrupted_prog, regex, replacement, extra_ids, last_id):
    positions = [m.span() for m in re.finditer(regex, corrupted_prog)]

    if len(positions) > 1:
        to_corrupt = rng.randint(len(positions))
    elif len(positions) == 1:
        to_corrupt = 0
    else:
        return corrupted_prog, None, None, extra_ids, last_id

    corrupted_prog = corrupted_prog[:positions[to_corrupt][0]] + \
        replacement + corrupted_prog[positions[to_corrupt][1]:]

    fix = isolate_line(org_prog, positions[to_corrupt][0])
    line = extract_line_number(org_prog, positions[to_corrupt][0])

    return corrupted_prog, fix, line, extra_ids, last_id


def easy_mutate(rng, org_prog, corrupted_prog, extra_ids, last_id):
    actions = {'replace while with for': ("_<keyword>_while", "_<keyword>_for"),
               'replace for with while': ("_<keyword>_for", "_<keyword>_while"),
               'NULL to null': ("_<keyword>_NULL", "_<keyword>_null"),
               'sizeof to size of': ("_<keyword>_sizeof", "")
              }

    action = rng.choice(actions.keys())

    # Make a new id for 'size'
    try:
        size_id = extra_ids['size']
    except KeyError:
        size_id = last_id + 1
        last_id += 1

    # Make a new id for 'of'
    try:
        of_id = extra_ids['of']
    except KeyError:
        of_id = last_id + 1
        last_id += 1

    if action == 'sizeof to size of':
        replace_with = ('_<id>_%d@ _<id>_%d@' % (size_id, of_id))
        last_id += 2
    else:
        replace_with = actions[action][1]

    return find_and_replace(rng, org_prog, corrupted_prog, actions[action][0], replace_with, extra_ids, last_id)


def get_last_id(tokens):
    result = None

    for token in tokens.split():
        if '_<id>_' in token:
            this_id = int(token.lstrip('_<id>_').rstrip('@'))

            if (result is None) or result < this_id:
                result = this_id

    return result


def id_mutate(rng, prog, max_num_mutations, num_mutated_progs, exact=False, name_dict=None):
    assert max_num_mutations > 0 and num_mutated_progs > 0, "Invalid argument(s) supplied to the function token_mutate"
    
    corrupt_fix_pair = []

    for _ in range(num_mutated_progs):
        corrupted = []
        fixes = []
        temp = []
        tokens = prog

        if exact:
            num_mutations = max_num_mutations
        else:
            num_mutations = rng.choice(range(max_num_mutations)) + 1
        deleted_vars = []
        mutation_count = 0

        fix = None
        fix_line = None

        for _ in range(num_mutations):
            # Step 2: Induce _[ONE]_ mutation, removing empty lines and shifting program if required
            try:
                mutated, this_fix, _ = undeclare_variable(
                    rng, tokens, tokens, deleted_vars, name_dict)
                mutation_count += 1

            # Couldn't delete anything
            except NothingToMutateException:
                break

            # Insertion or something failed
            except FailedToMutateException:
                raise
                continue

            # Deleted something that can't be fixed (all uses gone from the program)
            else:
                # REPLACE! program with mutated version
                tokens = mutated

                if not fix_ids_are_in_program(mutated, this_fix):
                    # Discarding previous fix: all uses are gone
                    continue

                fix_line = this_fix

            if fix_line != None:
                corrupted.append(tokens)
                fixes.append(fix_line)

        fixes.reverse()
        for i, c in enumerate(corrupted):
            temp.append((c, fixes[len(corrupted)-1-i:]))

        corrupt_fix_pair.append(temp)

    return corrupt_fix_pair
