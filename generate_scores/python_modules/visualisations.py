from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch


def convolution_matrix(labels, pred, categ):
    
    """Plot the confusion matrix

    Args:
        labels (array-like): True labels.
        pred (array-like): Predicted labels.
        categ (array-like): Categories.

    Returns:
        None
    """
    
    
    # Print Performances
    accuracy = accuracy_score(labels, pred)*100
    f1_weighted = f1_score(labels, pred, average='weighted')
    f1_macro = f1_score(labels, pred, average='macro')

    print(f"Accuracy: {accuracy:.4f}%, F1 Weighted Score: {f1_weighted:.4f}, F1 Macro score: {f1_macro:.4f}")


    #create the confusion matrix
    cm = confusion_matrix(labels, pred)
    dfcm = pd.DataFrame(cm, categ, categ)

    
    # plot the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(dfcm, cmap='viridis', interpolation='nearest', vmin=10, vmax=np.mean(cm))
    for i in range(len(dfcm.index)):
        for j in range(len(dfcm.columns)):
            plt.text(j, i, str(dfcm.iloc[i, j]), ha='center', va='center', color='white', fontsize=12)
    plt.xticks(range(len(dfcm.columns)), dfcm.columns)
    plt.yticks(range(len(dfcm.index)), dfcm.index)
    cbar = plt.colorbar()
    cbar.set_label('Values')
    plt.show()
    return 



def feature_importances_neural(model_neural, columns_data, smoothness=0, pos=range(0,599), plot=1):
    """Plot feature importances for different models and return the smoothed scores.

    Args:
        model_neural (nn.Module): Neural network model.
        columns_data (pd.DataFrame): columns informations of the dataset.
        smoothness (int): size of the window of the moving average filter (default is 0).
        pos (range or list): Range or list of positions in the sequence (default is range(0,599)).
        plot (boolean): 1 if we want to plot the results, 0 if we don't.

    Returns:
        pd.DataFrame: relevance score of each position.
    """
    
    # Empty dataframe to be filled
    scores = pd.DataFrame(index=[f'pos_{i}' for i in pos])  
    
    # Computing the features importance in the neural network
    eye = torch.tensor(np.eye(len(columns_data)), dtype=torch.float32)
    zero = torch.tensor(np.zeros(len(columns_data)), dtype=torch.float32)
    
    if torch.cuda.is_available():
        model_neural = model_neural.cpu()
        
    neural_impo = model_neural(eye) - model_neural(zero)
    
    if torch.cuda.is_available():
        neural_impo = neural_impo.cpu()
        model_neural = model_neural.cuda()
        
    
    # Reshaping neural_impo into a dataframe
    neural_impo = neural_impo.detach().numpy()
    neural_impo = np.mean(neural_impo, axis=1)
    neural_impo = neural_impo.reshape((len(neural_impo), 1))
    neural_impo = pd.DataFrame(neural_impo.T, columns=columns_data)
    
    # Loop for every position, sum the features importances (for each method) of every position 
    for i in pos:
        selected_columns = neural_impo.filter(regex=f'_{i}_')
        scores.loc[f'pos_{i}', 'Neural_scores'] = np.linalg.norm(np.abs(selected_columns), axis=1)
        
    # Standardize the scores
    scores_standardize = pd.DataFrame(StandardScaler().fit_transform(scores), columns=scores.columns, index=scores.index)
    scores_series = scores_standardize['Neural_scores']

    if smoothness > 1:
        # Add zero padding
        padded_scores_series = pd.concat([pd.Series([0] * (smoothness - 1)), 
                                          scores_series, 
                                          pd.Series([0] * (smoothness - 1))], ignore_index=True)
        # Calculate the moving average
        smoothed_scores_series = padded_scores_series.rolling(window=smoothness, min_periods=1, center=True).mean()
    
        
        # Remove the zero padding from the smoothed scores
        smoothed_scores_series = smoothed_scores_series[smoothness - 1:len(smoothed_scores_series) - (smoothness - 1)] 
        scores_smoothed = pd.DataFrame({'Neural_scores': smoothed_scores_series.values}, index=scores.index)
        scores_standardize = pd.DataFrame(StandardScaler().fit_transform(scores_smoothed), columns=scores.columns, index=scores.index)

    
    if plot:
        # Plot the smoothed scores
        plt.scatter(pos, scores_standardize['Neural_scores'], label='Neural_scores', color='blue')
        plt.title('Position importances')
        plt.xlabel('Positions')
        plt.ylabel('Scores')
        plt.legend()
        plt.show()
    
    return scores_standardize





def Modify_PDB_file(input_file, scores):
    """Create a copy of a .pdb file replacing the temperature factor (B-factor) by the relevance score for each position.

    Args:
        input_file (string): path to the input .pdb file.
        scores (list): list of the relevance scores.

    Returns:
        None
    """
    n = 0
    output_file =f"output/pdb_files/{input_file[:-4]}_scores.pdb"
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.startswith('ATOM'):
                # Get the position in the sequence
                position = int(line[22:26])

                if 1 <= position <= len(scores):
                    # Replace the B-factor with the corresponding value from the list
                    line = line[:60] + "{:6.2f}".format(scores[position - 1]) + line[66:]
                else:
                    # If the position is out of range, replace the B-factor with 0
                    line = line[:60] + "{:6.2f}".format(0) + line[66:]

                n += 1

            f_out.write(line)

    print(f"{output_file} has been created with {n} values modified")


