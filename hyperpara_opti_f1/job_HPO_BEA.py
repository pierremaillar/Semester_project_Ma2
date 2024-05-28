import numpy as np
import pandas as pd
import time

from python_modules.create_batch import *
from python_modules.datatreatment import *
from python_modules.neuralnet import *
from python_modules.visualisations import *
from python_modules.nn_hyperpara_opti import *




path_to_dataset ="../dataset_hsp70_tax/dataset_hsp70_tax.csv" 
hsp70 = importing_data(path_to_dataset)


level3, level3_categ= get_data(hsp70, 2, 3,Use_Others=False)
level3=encode01(level3)
level3=category_to_int(level3,level3_categ)

param_grid = {
        'layer_dim': [32,64,128,256],
        'number_hidden_layer': [2,3,4,5],
        'dropout_prob': [0.2,0.4,0.6,0.8],
        'l2_regu': [1e-05],
        'weight_decay': [0.0001],
        'learning_rate':[0.0001,0.001],
        'batch_size':[64,512,2048],
        'num_epochs':[5,10,15]
        }


nbr_batch = 5


train, train_label, test, test_label, val, val_label=split_dataset(level3, 0.8, 0.1, 0.1)

output_dim = 3
best_params_nn,_,_ = optimize_hyperparameters_nn(train, train_label, val, val_label, nbr_batch,output_dim, param_grid)



print("Job finished")