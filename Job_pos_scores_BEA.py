import numpy as np
import pandas as pd
import time
from create_batch import *
from datatreatment import *
from neuralnet import *
from visualisations import *


path_to_dataset ="dataset_hsp70_tax/dataset_hsp70_tax.csv" 
hsp70 = importing_data(path_to_dataset)


level3, level3_categ= get_data(hsp70, 2, 3,Use_Others=False)
level3=encode01(level3)
level3=category_to_int(level3,level3_categ)
columns_info = level3.drop(level3.columns[0], axis=1).columns
positions_to_keep =range(0,599)


best_params_nn = {'layer_dim': 128, 'number_hidden_layer': 2, 'dropout_prob': 0.8, 'l2_regu': 1e-05, 'weight_decay': 0.0001, 'learning_rate': 0.0001, 'batch_size': 1024, 'num_epochs': 10}




layer_dim = best_params_nn['layer_dim']
number_hidden_layer = best_params_nn['number_hidden_layer']
dropout_prob = best_params_nn['dropout_prob']
l2_regu = best_params_nn['l2_regu']
weight_decay = best_params_nn['weight_decay']
learning_rate = best_params_nn['learning_rate']
batch_size = best_params_nn['batch_size']
num_epochs = best_params_nn['num_epochs']


output_dim = 3
tic = time.time()
dfs = []

for i in range(30):
    
    train, train_label, test, test_label, val, val_label=split_dataset(level3, 0.8, 0.1, 0.1)
    input_dim = train.shape[1]
    
    model_neural = ModelClassification(input_dim, output_dim, layer_dim, number_hidden_layer, dropout_prob, l2_regu)
    optimizer = torch.optim.Adam(model_neural.parameters(), lr = learning_rate, weight_decay=weight_decay)
    train_model(model_neural, num_epochs, train, train_label, test, test_label, optimizer, batch_size)
    
    dfs.append(feature_importances_neural(model_neural, columns_info, smoothness = 0, pos =positions_to_keep, plot = 0))
    t = time.time() - tic
    print(f"Got to iteration {i+1} in {t} seconds")
    
    

arrays = [df.to_numpy() for df in dfs]

stacked_array = np.stack(arrays, axis=0)
mean_values = np.mean(stacked_array, axis=0)
std_values = np.std(stacked_array, axis=0)




plt.figure(figsize=(50, 6))
plt.errorbar(positions_to_keep, mean_values[:, 0], std_values[:, 0], capsize=4, color="red", fmt="o", markersize=4)

plt.xlabel('Positions')
plt.ylabel('Scores')
plt.title('Position importances')


num_ticks = len(positions_to_keep)//15
xticks_indices = np.linspace(0, len(positions_to_keep) - 1, num_ticks, dtype=int)
plt.xticks(np.array(positions_to_keep)[xticks_indices])

plt.savefig('Postitionimportances_opti_BEA.png', dpi=200)




np.savetxt('mean_BEA.txt', mean_values)
np.savetxt('std_BEA.txt', std_values)


print("Job finished")