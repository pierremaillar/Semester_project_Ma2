# Semester_project: Illustration of neural network interpretation

## General description
This project focuses on the analysis of amino acid sequences within Hsp70 proteins with dual objectives. The primary goal is the prediction of the taxonomic group to which the proteins belong. Additionally, we aim to identify critical positions within the sequence that are pivotal for protein classification.

To achieve these objectives, we use both a Random Forest algorithm and a Neural Network for the classification.
## Installations required
- pytorch
- panda
- numpy
- scikit-learn
- plotly
- matplotlib
## Files description
### hyperpara_opti_f1: Optimize the hyperparameters using the f1 score as a criterion
- **HPO_BEA.run, HPO_EUK.run, HPO_BAC.run, HPO_task30.run,** : to submit a job to Izar in order to optimize hyperparameters.
### hyperpara_opti_consistency: Optimize the hyperparameters using the consistency score as a criterion
- **HPO_cons_BEA.run, HPO_cons_EUK.run, HPO_cons_BAC.run, HPO_cons_task30.run,** : to submit a job to Izar in order to optimize hyperparameters.
### generate_scores: to generate relevant scores after hyperpara selection
- **input_pdb** : folder containing the .pdb files to modify adding the relevant scores
- **scores_BEA.run, scores_EUK.run, scores_BAC.run, scores_task30.run,** : to submit a job to Izar in order to calculate the relevant score for each task.
###
### Python modules (function)
- **datatreatment.py** : import and formatting the dataset.
- **create_batch.py** : create training set, validation set and test set.
- **neuralnet.py, nn_hyperpara_opti.py, nn_hyperpara_opti_consistency.py** : contains neural network architecture, its training and its hyperparameter optimization function.
- **visualisations.py** : plot the feature importances and other visualization plots.
### Requirements 
- **requirements.txt** : contains the required installations.


## Run computation on Izar
# 0. Connect to EPFL WIFI
If you're not on an EPFl network you must use a VPN

More info can be found here: https://www.epfl.ch/schools/sb/research/iphys/wp-content/uploads/2019/06/Welcome_IPHYS-IT.pdf

# 1. Setup

1. Make sure you have an account on izar. Throughout, I will be refering to your username when I write <gaspar username>

2. ssh into izar
```bash
ssh <gaspar username>@izar.epfl.ch
```

3. Clone this repository

```bash
git clone https://github.com/pierremaillar/Semester_project_Ma2.git
```

Note that downloading the repository from the command line requires a different authentication method than logging into the github web site. Your regular password should not work for downloading the repository. You have to create a token through the github password (or find another way to authenticate). The process is described on this web site:

https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic

#2. Install

```bash
cd Semester_project_Ma2
source setup.sh gasparusername
```

# 4. Submit a job to IZAR
 ```bash
sbatch run.sh
```
You can check up on it's progress using this command:
 ```bash
squeue -u gasparusername
```

# 5. Downloading results to local computer
If you are connected to izar run the following command:
```bash
exit
```
Then run:
```bash
scp -r gasparusername@izar.epfl.ch:/home/gasparusername/Protein_design/results "path\to\local\storage_file"

```
To get the path, similar to pdb file right click on folder where you want the results to be downloaded and select "copy as path".

# 6. Updating the pipline
In case there are updates. The pipline can be updated as follows:
```bash
ssh <gaspar username>@izar.epfl.ch
cd Protein_design
git pull
```

## Author
Pierre Maillard: pierre.maillard@epfl.ch

