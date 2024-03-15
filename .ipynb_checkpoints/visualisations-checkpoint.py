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



def feature_importances(model_rdf, model_neural, mutual_info, data_df, smoothness = 30, nbr_pos = 599, plot = 1):
    """Plot feature importances for different models and return the smoothed scores.

    Args:
        model_rdf (RandomForestClassifier): Random Forest model.
        model_neural (nn.Module): Neural network model.
        mutual_info (numpy array): Mutual information scores.
        data_df (pd.DataFrame): Input data.
        smoothness (int): Smoothness factor for plotting (default is 30).
        nbr_pos (int): Number of positions in the sequence(default is 599).
        plot(boolean): 1 if we want to plot de results, 0 if we dont.

    Returns:
        pd.DataFrame: Smoothed scores for feature importances of each positions.
    """
    #drop the first columns containing the labels
    columns_data = data_df.drop(data_df.columns[0], axis=1).columns
    #empty dataframe to be filled
    scores = pd.DataFrame(index=[f'pos_{i}' for i in range(1,nbr_pos+1)])  
    
    #computing the features importance in the neural network
    eye = torch.tensor(np.eye(len(columns_data)), dtype=torch.float32)
    zero = torch.tensor(np.zeros(len(columns_data)), dtype=torch.float32)
    
    
    neural_impo = model_neural(eye)-model_neural(zero)
    if torch.cuda.is_available():
        neural_impo = neural_impo.cpu()
    
    #reshaping neural_impo into a dataframe
    neural_impo = neural_impo.detach().numpy()
    neural_impo = np.mean(neural_impo,axis=1)
    neural_impo = neural_impo.reshape((len(neural_impo),1))
    neural_impo = pd.DataFrame(neural_impo.T, columns=columns_data)
    
    
    #reshaping rdf_impo into a dataframe
    rdf_impo = model_rdf.feature_importances_
    rdf_impo = rdf_impo.reshape((len(rdf_impo),1))
    rdf_impo = pd.DataFrame(rdf_impo.T, columns=columns_data)

    #reshaping mutual_info into a dataframe
    mutual_info = mutual_info.reshape((len(mutual_info),1))
    mutual_df = pd.DataFrame(mutual_info.T, columns=columns_data)

    #loop for every positions, sum the features importances (for each methods) of every positions 
    for i in range(1,nbr_pos+1):
        selected_columns = neural_impo.filter(regex=f'_{i}_')
        scores.loc[f'pos_{i}','Neural_scores'] = np.linalg.norm(np.abs(selected_columns), axis=1)
        
        
        selected_columns = mutual_df.filter(regex=f'_{i}_')
        scores.loc[f'pos_{i}','Mutual_scores'] = np.linalg.norm(np.abs(selected_columns), axis=1)
        
        
       
        selected_columns = rdf_impo.filter(regex=f'_{i}_')
        scores.loc[f'pos_{i}','rdf_scores'] = np.linalg.norm(np.abs(selected_columns),axis = 1)
     
    
    #standardize the scores
    scores_standardize = pd.DataFrame(StandardScaler().fit_transform(scores), columns=scores.columns, index=scores.index)
    
    
    #smooth the scores with a moving average
    scores_smoothed = scores_standardize.rolling(window=smoothness, min_periods=1,center=True).mean()
    scores_smoothed = pd.DataFrame(StandardScaler().fit_transform(scores_smoothed), columns=scores_smoothed.columns, index=scores_smoothed.index)
    if plot:
        #plots
        plt.plot(range(1,nbr_pos+1),scores_smoothed['Neural_scores'], label='Neural_scores', color='blue')
        plt.plot(range(1,nbr_pos+1),scores_smoothed['Mutual_scores'], label='Mutual_scores', color='green')
        plt.plot(range(1,nbr_pos+1),scores_smoothed['rdf_scores'], label='rdf_scores', color='red')
        plt.title('Position importances')
        plt.xlabel('Positions')
        plt.ylabel('Scores')
        plt.legend()
        plt.show()
    
    return scores_smoothed
