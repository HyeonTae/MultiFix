echo 'Downloading DeepFix raw dataset...'
wget https://www.cse.iitk.ac.in/users/karkare/prutor/prutor-deepfix-09-12-2017.zip -P data/deepfix_raw_data
cd data/deepfix_raw_data
unzip prutor-deepfix-09-12-2017.zip
mv prutor-deepfix-09-12-2017/* ./
rm -rf prutor-deepfix-09-12-2017 prutor-deepfix-09-12-2017.zip
gunzip prutor-deepfix-09-12-2017.db.gz
mv prutor-deepfix-09-12-2017.db dataset.db
cd ../..

echo 'Preprocessing DeepFix dataset...'
export PYTHONPATH=.
python data_processing/DeepFix/preprocess.py
python data_processing/DeepFix/test_data_generator.py

echo 'Downloading DrRepair dataset...'
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--auto-corrupt--orig-deepfix.zip -P data_processing
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--auto-corrupt--codeforce--deepfix-style.zip -P data_processing
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--auto-corrupt--codeforce--spoc-style.zip -P data_processing
wget https://nlp.stanford.edu/projects/myasu/DrRepair/data/err-data-compiler--orig-spoc.zip -P data_processing

cd data_processing
unzip err-data-compiler--auto-corrupt--orig-deepfix.zip
unzip err-data-compiler--auto-corrupt--codeforce--deepfix-style.zip
unzip err-data-compiler--auto-corrupt--codeforce--spoc-style.zip
unzip err-data-compiler--orig-spoc.zip
rm err-data-compiler--auto-corrupt--orig-deepfix.zip
rm err-data-compiler--auto-corrupt--codeforce--deepfix-style.zip
rm err-data-compiler--auto-corrupt--codeforce--spoc-style.zip
rm err-data-compiler--orig-spoc.zip
mv err-data-compiler--auto-corrupt--orig-deepfix DrRepair_deepfix
mv err-data-compiler--orig-spoc DrRepair_spoc
mv err-data-compiler--auto-corrupt--codeforce--deepfix-style DrRepair_codeforce_deepfix_style
mv err-data-compiler--auto-corrupt--codeforce--spoc-style DrRepair_codeforce_spoc_style

cd DrRepair_deepfix/err-data-compiler--auto-corrupt--orig-deepfix
mv bin0/* ./
mv bin1/* ./
mv bin2/* ./
mv bin3/* ./
mv bin4/* ./
rm -rf bin0 bin1 bin2 bin3 bin4
cd ../../..

cd DrRepair_spoc/err-data-compiler--orig-spoc
mv s1/* ./
mv s2/* ./
mv s3/* ./
mv s4/* ./
mv s5/* ./
rm -rf s1 s2 s3 s4 s5 reset.sh

#echo 'Vocab generation...'
#python data_processing/DrRepair_spoc/vocab_generator.py
#python data_processing/DrRepair_codeforce_spoc_style/vocab_generator.py
#python data_processing/DrRepair_codeforce_deepfix_style/vocab_generator.py

echo 'DeepFix Data generation...'
python data_processing/DeepFix/data_generator.py
mkdir data/DeepFix
cat data/DeepFix/ids/data_train.txt data/DeepFix/typo/data_train.txt > data/DeepFix/data_train.txt
cat data/DeepFix/ids/data_val.txt data/DeepFix/typo/data_val.txt > data/DeepFix/data_val.txt
shuf -o data/DeepFix/data_train.txt data/DeepFix/data_train.txt
shuf -o data/DeepFix/data_val.txt data/DeepFix/data_val.txt
rm -rf data/DeepFix/ids
rm -rf data/DeepFix/typo

echo 'DrRepair Data generation...'
python data_processing/DrRepair_deepfix/data_generator.py
python data_processing/DrRepair_spoc/data_generator.py
python data_processing/DrRepair_codeforce_deepfix_style/data_generator.py
python data_processing/DrRepair_codeforce_spoc_style/data_generator.py
