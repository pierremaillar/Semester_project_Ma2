import numpy as np
import pandas as pd
import time
from python_modules.create_batch import *
from python_modules.datatreatment import *
from python_modules.neuralnet import *
from python_modules.visualisations import *


path_to_dataset ="../dataset_hsp70_tax/dataset_hsp70_tax.csv" 
hsp70 = importing_data(path_to_dataset)


level3, level3_categ= get_data(hsp70, 3, 4,"Eukaryota",Use_Others=True)
level3=encode01(level3)
level3=category_to_int(level3,level3_categ)
columns_info = level3.drop(level3.columns[0], axis=1).columns
positions_to_keep =range(0,599)


#best f1 score
#best_params_nn = {'layer_dim': 128, 'number_hidden_layer': 3, 'dropout_prob': 0.2, 'l2_regu': 1e-05, 'weight_decay': 0.0001, 'learning_rate': 0.001, 'batch_size': 64, 'num_epochs': 15}
#best consitency
best_params_nn = {'layer_dim': 32, 'number_hidden_layer': 3, 'dropout_prob': 0.7, 'l2_regu': 1e-05, 'weight_decay': 0.0001, 'learning_rate': 0.0001, 'batch_size': 64, 'num_epochs': 8}


layer_dim = best_params_nn['layer_dim']
number_hidden_layer = best_params_nn['number_hidden_layer']
dropout_prob = best_params_nn['dropout_prob']
l2_regu = best_params_nn['l2_regu']
weight_decay = best_params_nn['weight_decay']
learning_rate = best_params_nn['learning_rate']
batch_size = best_params_nn['batch_size']
num_epochs = best_params_nn['num_epochs']


output_dim = 4
tic = time.time()
dfs = []
train, train_label, test, test_label, val, val_label=split_dataset(level3, 0.8, 0.1, 0.1)
input_dim = train.shape[1]

model_neural = ModelClassification(input_dim, output_dim, layer_dim, number_hidden_layer, dropout_prob, l2_regu)
optimizer = torch.optim.Adam(model_neural.parameters(), lr = learning_rate, weight_decay=weight_decay)

for i in range(30):
    train_model(model_neural, num_epochs, train, train_label, test, test_label, optimizer, batch_size) 
    dfs.append(feature_importances_neural(model_neural, columns_info, smoothness = 0, pos =positions_to_keep, plot = 0))
    t = time.time() - tic
    print(f"Got to iteration {i+1} in {t} seconds")
    
    

arrays = [df.to_numpy() for df in dfs]

stacked_array = np.stack(arrays, axis=0)
mean_values = np.mean(stacked_array, axis=0)
std_values = np.std(stacked_array, axis=0)
print(f"consistency score: {np.mean(std_values)}")

#save data
np.savetxt('output/mean_EUK.txt', mean_values)
np.savetxt('output/std_EUK.txt', std_values)


scores = pd.DataFrame({'mean': mean_values.flatten(), 'std': std_values.flatten()})
norm = plt.Normalize(scores['mean'].min(), scores['mean'].max())


plt.figure(figsize=(60, 6))
plt.style.use('dark_background')
plt.xlabel('Positions', color='white')
plt.ylabel('Scores', color='white')
plt.title('Relevant scores EUK', color='white')


colors = plt.cm.bwr(norm(scores['mean']))
for i in range(len(scores["mean"])):
    plt.errorbar(i, scores["mean"][i], scores["std"][i], capsize=4, color=colors[i], fmt="o", markersize=4)

num_ticks = len(scores["mean"]) // 10
xticks_indices = np.linspace(0, len(scores["mean"]) - 1, num_ticks, dtype=int)
plt.xticks(np.arange(len(scores["mean"]))[xticks_indices], color='white')
plt.yticks(color='white')



plt.savefig('output/Postitionrelevance_EUK.png', dpi=200)


#modify pdb file
smoothness = 30
scores['mean'] = scores['mean'].rolling(window=smoothness, min_periods=1, center=True).mean()
relevant_scores = scores['mean']


#name of the pdb file
input_pdb = "5nro.pdb"

Modify_PDB_file(input_pdb, relevant_scores, "EUK")


print("Job finished")