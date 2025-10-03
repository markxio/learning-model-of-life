#!/bin/bash

apt-get update
apt-get install git
apt install python3.12
apt install python3.12-venv

apt-get install vim
apt-get install wget
apt-get install gzip

apt-get install python3-dev # python header files 

python3.12 -m venv venv
source venv/bin/activate

# if "cant find torch" or similar, use --no-build-isolation
pip install ... --no-build-isolation

git clone https://github.com/Zymrael/savanna.git
cd savanna
make setup-env

# prepare/preprocess data
pip install numpy tqdm

apt install g++ # needed?

# flash_attn install fails due to not finding torch
# solution: https://github.com/Dao-AILab/flash-attention/issues/1421#issuecomment-2575547768
pip install psutil
pip install numpy
pip install ninja
pip install hatchling
pip install --global-option="build_ext" --global-option="-j32" flash-attn==2.6.3 --no-build-isolation

# up to here tested

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
