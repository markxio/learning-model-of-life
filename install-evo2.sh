#!/bin/bash

apt-get update
apt-get install git
apt install python3.11
apt install python3.11-venv
python3.11 -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
pip install pipenv
pipenv install "setuptools==71.1.0"
pipenv sync --system --dev

apt-get install python3.11-dev

git clone --recurse-submodules https://github.com/markxio/evo2
cd evo2

pip install --use-pep517 --no-build-isolation .

# I was getting KeyError: 'recipe', 
# but after pip uninstall transformer-engine (removing 2.1.0) 
# and then running pip install transformer_engine[pytorch]==1.13
# it was fixed for me.
pip uninstall transformer_engine
pip install transformer_engine[pytorch]==1.13

python ./test/test_evo2.py --model_name evo2_7b
#python ./test/test_evo2.py --model_name evo2_40b
