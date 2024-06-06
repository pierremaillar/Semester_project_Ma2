#!/bin/bash

user="${1}"
if [[ "$user" == "" ]]; then
    echo -e "\e[93;1mMust specify username\e[0m"
    exit 1
fi

pip install --upgrade pip

# Miniconda3

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
echo -e "\e[91;1m│ Install conda to /home/$user/miniconda3 │\e[0m"
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
conda init
conda update conda

conda env create -f env_pytorch
conda activate env_pytorch

pip install -q --upgrade --user -r requirements.txt
