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
