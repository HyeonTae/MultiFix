import os
import collections
import regex as re
import copy
from util.helpers import get_lines, recompose_program
from util.tokenizer import Tokenizer, UnexpectedTokenException, EmptyProgramException

Token = collections.namedtuple('Token', ['typ', 'value', 'line', 'column'])


class C_Tokenizer(Tokenizer):
    _keywords = set(['auto', 'break', 'case', 'const', 'continue', 'default',
                 'do', 'else', 'enum', 'extern', 'for', 'goto', 'if',
                 'register', 'return', 'sizeof', 'static', 'switch',
                 'typedef', 'void', 'volatile', 'while', 'EOF', 'NULL', 'endl',
                 'null', 'struct', 'union'] + \
                 [
                  'alignas', 'alignof', 'and', 'and_eq', 'asm', 'atomic_cancel',
                  'atomic_commit', 'atomic_noexcept', 'auto', 'bitand', 'bitor',
                  'break', 'case', 'catch',
                  'class', 'co_await', 'co_return', 'co_yield', 'compl', 'concept', 'const',
                  'const_cast', 'consteval', 'constexpr', 'continue', 'decltype', 'default',
                  'delete', 'do', 'dynamic_cast', 'else', 'enum', 'explicit',
                  'export', 'extern', 'false', 'for', 'friend', 'goto', 'if',
                  'import', 'inline', 'module', 'mutable', 'namespace', 'new',
                  'noexcept', 'not', 'not_eq', 'nullptr', 'operator', 'or', 'or_eq',
                  'private', 'protected', 'public', 'reflexpr', 'register', 'reinterpret_cast',
                  'requires', 'return', 'sizeof', 'static', 'static_assert',
                  'static_cast', 'struct', 'switch', 'synchronized', 'template', 'this',
                  'thread_local', 'throw', 'true', 'try', 'typedef', 'typeid', 'typename',
                  'union', 'using', 'virtual', 'void', 'volatile',
                  'while', 'xor', 'xor_eq',
                 ])
    _includes = set(['stdio.h', 'stdlib.h', 'string.h', 'math.h', 'malloc.h',
                 'stdbool.h', 'cstdio', 'cstdio.h', 'iostream', 'conio.h'])
    _includes.update(["<" +inc+ ">" for inc in _includes] + ["<string>", "<bits/stdc++.h>"])

    _calls = set(['printf', 'scanf', 'cin', 'cout', 'clrscr', 'getch', 'strlen',
              'gets', 'fgets', 'getchar', 'main', 'malloc', 'calloc', 'free', 'sort'] + \
               open(os.path.dirname(os.path.abspath(__file__)) + "/cpp_functions.txt",'r').read().split('\n')) #ADDED
    _types = set(['char', 'double', 'float', 'int', 'long', 'short', 'unsigned'] + ['signed', 'char16_t', 'char32_t', 'char8_t', 'wchar_t', 'string', 'bool'])

    _ops = set('(|)|[|]|{|}|->|<<|>>|**|&&|--|++|-=|+=|*=|&=|%=|/=|==|<=|>=|!=|-|<|>|~|!|%|^|&|*|/|+|=|?|.|,|:|;|#'.split('|') + ['||','|=','|'])

    def _escape(self, string):
        return repr(string)[1:-1]

    def _tokenize_code(self, code):
        keywords = {'IF', 'THEN', 'ENDIF', 'FOR', 'NEXT', 'GOSUB', 'RETURN'}
        token_specification = [
            ('comment',
             r'\/\*(?:[^*]|\*(?!\/))*\*\/|\/\*([^*]|\*(?!\/))*\*?|\/\/[^\n]*'),
            ('directive', r'#\w+'),
            ('string', r'"(?:[^"\n]|\\")*"'),
            ('string_continue', r'"\s[^\s\n=\(\){}\[\];<>"\,\-\+\?\:\|\& (return)]*|[^\s\n=\(\){}\[\];<>"\,\-\+\?\:\|\& (return)]+\s"'),
            ('char', r"'(?:[^'\n]|\\')*'"),
            ('char_continue', r"'\s[^\s\n=\(\){}\[\];<>'\,\-\+\?\:\|\& (return)]*|[^\s\n=\(\){}\[\];<>'\,\-\+\?\:\|\& (return)]+\s'"),
            ('number',  r'0[x|X]\w*|[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?L*U*l*u*'),
            ('include',  r'(?<=\#include) *.*'),
            ('op',
             r'\(|\)|\[|\]|{|}|->|<<|>>|\*\*|\|\||&&|--|\+\+|[-+*|&%\/=<>!]=|[-<>~!%^&*\/+=?|.,:;#]'),
            ('name',  r'[_A-Za-z]\w*|\$+[_A-Za-z]*\w*'),
            ('whitespace',  r'\s+'),
            ('nl', r'\\\n?'),
            ('MISMATCH', r'.'),            # Any other character
        ]
        tok_regex = '|'.join('(?P<%s>%s)' %
                             pair for pair in token_specification)
        line_num = 1
        line_start = 0
        for mo in re.finditer(tok_regex, code):
            kind = mo.lastgroup
            value = mo.group(kind)
            if kind == 'NEWLINE':
                line_start = mo.end()
                line_num += 1
            elif kind == 'SKIP':
                pass
            elif kind == 'MISMATCH':
                yield UnexpectedTokenException('%r unexpected on line %d' % (value, line_num))
            else:
                if kind == 'ID' and value in keywords:
                    kind = value
                column = mo.start() - line_start
                yield Token(kind, value, line_num, column)

    def _sanitize_brackets(self, tokens_string):
        lines = get_lines(tokens_string)

        if len(lines) == 1:
            raise EmptyProgramException(tokens_string)

        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]

            if line.strip() == '_<op>_}' or line.strip() == '_<op>_} _<op>_}' \
               or line.strip() == '_<op>_} _<op>_} _<op>_}' or line.strip() == '_<op>_} _<op>_;' \
               or line.strip() == '_<op>_} _<op>_} _<op>_} _<op>_}' \
               or line.strip() == '_<op>_{' \
               or line.strip() == '_<op>_{ _<op>_{':
                if i > 0:
                    lines[i - 1] += ' ' + line.strip()
                    lines[i] = ''
                else:
                    # can't handle this case!
                    return ''

        # Remove empty lines
        for i in range(len(lines) - 1, -1, -1):
            if lines[i] == '':
                del lines[i]

        for line in lines:
            assert(lines[i].strip() != '')

        return recompose_program(lines)

    def tokenize(self, code, prev_name_dict=None, prev_number_dict=None, keep_format_specifiers=False, keep_names=True,
                 keep_literals=True):
        result = '0 ~ '

        if prev_name_dict is None:
            name_dict = {}
        else:
            name_dict = copy.deepcopy(prev_name_dict)

        if prev_number_dict is None:
            number_dict = {}
        else:
            number_dict = copy.deepcopy(prev_number_dict)

        names = ''
        line_count = 1
        name_sequence = []
        number_sequence = []

        regex = '%(d|i|f|c|s|u|g|G|e|p|llu|ll|ld|l|o|x|X)'
        isNewLine = True

        # Get the iterable
        my_gen = self._tokenize_code(code)

        while True:
            try:
                token = next(my_gen)
            except StopIteration:
                break

            if isinstance(token, Exception):
                return Exception
                #return '', '', ''

            type_ = str(token[0])
            value = str(token[1])

            #print("type_ = {}".format(type_))
            #print("value = {}".format(value))

            if value in self._keywords:
                result += '_<keyword>_' + self._escape(value) + ' '
                isNewLine = False

            elif type_ == 'include':
                result += '_<include>_' + self._escape(value).lstrip() + ' '
                isNewLine = False

            elif value in self._calls:
                result += '_<APIcall>_' + self._escape(value) + ' '
                isNewLine = False

            elif value in self._types:
                result += '_<type>_' + self._escape(value) + ' '
                isNewLine = False

            elif type_ == 'whitespace' and (('\n' in value) or ('\r' in value)):
                if isNewLine:
                    continue

                result += ' '.join(list(str(line_count))) + ' ~ '
                line_count += 1
                isNewLine = True

            elif type_ == 'whitespace' or type_ == 'comment' or type_ == 'nl':
                pass

            elif type_ == 'string_continue':
                result += '_<string_continue>_'  + ' '
                isNewLine = False

            elif 'string' in type_:
                matchObj = [m.group().strip()
                            for m in re.finditer(regex, value)]
                if matchObj and keep_format_specifiers:
                    for each in matchObj:
                        result += each + ' '
                else:
                    result += '_<string>_' + ' '
                isNewLine = False

            elif type_ == 'name':
                if keep_names:
                    if self._escape(value) not in name_dict:
                        name_dict[self._escape(value)] = str(
                            len(name_dict) + 1)

                    name_sequence.append(self._escape(value))
                    result += '_<id>_' + name_dict[self._escape(value)] + '@ '
                    names += '_<id>_' + name_dict[self._escape(value)] + '@ '
                else:
                    result += '_<id>_' + '@ '
                isNewLine = False

            elif type_ == 'number':
                if keep_literals:
                    if self._escape(value) not in number_dict:
                        number_dict[self._escape(value)] = str(
                            len(number_dict) + 1)

                    number_sequence.append(self._escape(value))
                    result += '_<number>_' + number_dict[self._escape(value)] + '# '
                else:
                    result += '_<number>_' + '# '
                isNewLine = False

            elif 'char' in type_ or value == '':
                result += '_<' + type_.lower() + '>_' + ' '
                isNewLine = False

            else:
                converted_value = self._escape(value)
                result += '_<' + type_ + '>_' + converted_value + ' '

                isNewLine = False

        result = result[:-1]
        names = names[:-1]

        if result.endswith('~'):
            idx = result.rfind('}')
            result = result[:idx + 1]

        return self._sanitize_brackets(result), name_dict, name_sequence, number_dict, number_sequence
