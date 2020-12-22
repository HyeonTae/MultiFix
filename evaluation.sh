mkdir log/test

export PYTHONPATH=.

echo 'Program repair evaluation...'

python main/test/ids_typo/test.py
python main/test/typo/test.py
python main/test/ids/test.py
python main/test/drrepair/test.py

# beam search
python main/test/ids_typo/test_beam_search.py
python main/test/typo/test_beam_search.py
python main/test/ids/test_beam_search.py
python main/test/drrepair/test_beam_search.py
