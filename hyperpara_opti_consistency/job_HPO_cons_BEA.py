import numpy as np
import pandas as pd
import time

from python_modules.create_batch import *
from python_modules.datatreatment import *
from python_modules.neuralnet import *
from python_modules.visualisations import *
from python_modules.nn_hyperpara_opti_consistency import *




path_to_dataset ="../dataset_hsp70_tax/dataset_hsp70_tax.csv" 
hsp70 = importing_data(path_to_dataset)


level3, level3_categ= get_data(hsp70, 2, 3,Use_Others=False)
level3=encode01(level3)
level3=category_to_int(level3,level3_categ)
columns_info = level3.drop(level3.columns[0], axis=1).columns

param_grid = {
        'layer_dim': [48,64,96,192],
        'number_hidden_layer': [2,3,4],
        'dropout_prob': [0.3,0.5,0.6,0.7],
        'l2_regu': [1e-05],
        'weight_decay': [0.0001],
        'learning_rate':[0.0001],
        'batch_size':[64,128,1024],
        'num_epochs':[5,7,10]
        }

positions_to_keep =range(0,599)
nbr_training = 5


train, train_label, test, test_label, val, val_label=split_dataset(level3, 0.8, 0.1, 0.1)

output_dim = 3
best_params_nn, mean_values, std_values = optimize_hyperparameters_nn_consistency(train, train_label, val, val_label, output_dim,columns_info,param_grid,nbr_training,positions_to_keep)




print("Job finished")