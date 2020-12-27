mkdir log/check_point
mkdir log/plot
mkdir log/pth

export PYTHONPATH=.

echo 'Model training...'
python main/train/deepfix/train.py
python main/train/drrepair_codeforce_deepfix_style/train.py
python main/train/drrepair_codeforce_spoc_style/train.py
python main/train/drrepair_deepfix/train.py
python main/train/drrepair_spoc/train.py
