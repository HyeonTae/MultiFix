mkdir log/check_point
mkdir log/plot
mkdir log/pth

export PYTHONPATH=.

echo 'Model training...'
python main/train/ids/train.py
python main/train/typo/train.py
python main/train/ids_typo/train.py
python main/train/drrepair/train.py
