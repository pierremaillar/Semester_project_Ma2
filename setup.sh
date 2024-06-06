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


# RFdiffusion

git clone https://github.com/RosettaCommons/RFdiffusion.git && cd RFdiffusion
mkdir models && cd models
wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt
wget http://files.ipd.uw.edu/pub/RFdiffusion/1befcb9b28e2f778f53d47f18b7597fa/RF_structure_prediction_weights.pt
cd ..


# SE3 transformer enviroment
conda env create -f env/SE3nv.yml
conda activate SE3nv
cd env/SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ../.. # change into the root directory of the repository
pip install -e . # install the rfdiffusion module from the root of the repository
conda deactivate
cd .. # return to Protein_design

#ProteinMPNN

git clone https://github.com/dauparas/ProteinMPNN.git

#Colabfold
conda create --name AF2seq python=3.9
conda activate AF2seq
conda install openmm==7.7.0 pdbfixer -c conda-forge
pip install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1
pip install "colabfold[alphafold] @ git+https://github.com/sokrypton/ColabFold"
conda install -c conda-forge libstdcxx-ng==9.3.0
pip install jax==0.4.9 
pip install numpy==1.24.0
conda deactivate

#Loop through scripts and replace "rahi" by your user name
for f in ~/Protein_design/*.sh
do
    echo "Replacing rahi for your username $user in $f "
    sed -i "s|rahi|$user|g" $f
done

