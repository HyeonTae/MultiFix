mkdir log/test

export PYTHONPATH=.

echo 'Program repair evaluation...'

python main/test/deepfix/test.py
python main/test/drrepair_deepfix/test.py

# beam search
python main/test/deepfix/test_beam_search.py
python main/test/drrepair_deepfix/test_beam_search.py
