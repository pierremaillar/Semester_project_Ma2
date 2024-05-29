import itertools
import time
import numpy as np
from python_modules.neuralnet import ModelClassification
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from python_modules.visualisations import *

def predict(model, X, device):
    """Generate predictions using a neural network model.

    Args:
        model (ModelClassification): The trained neural network model.
        X (numpy array): Input data for predictions.
        device (torch.device): The device (cuda or cpu) on which the model should run.

    Returns:
        numpy array: Predicted labels.
    """
    # Convert input data to PyTorch tensor and move to the specified device
    inputs = torch.Tensor(X).to(device)

    # Set the model to evaluation mode and make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    
    # Convert predictions to NumPy array
    predictions = outputs.argmax(dim=1).cpu().numpy()

    return predictions

def cross_val_pos_consistency(hyperparam_dict, train_inputs, train_labels , val_inputs, val_labels,output_dim,columns_info, device, nbr_training, pos = range(0,600)):
    """Perform cross-validation for a neural network model many times. (nbr_training times)

    Args:
        hyperparam_dict (dict): Dictionary mapping hyperparameter names to values.
        train_loader (DataLoader): DataLoader for training data.
        val_inputs (numpy array): Validation data.
        val_labels (numpy array): True labels for validation data.
        output_dim (int): Dimension of the output (number of classes).
        columns_data (pd.DataFrame): columns informations of the dataset.
        device (torch.device): The device (cuda or cpu) on which the model should run.
        nbr_training (int): Number of times the neural net will be retrained to compute de std of the scores.

    Returns:
        tuple of floats: Mean and standard deviation of the F1 score. Mean and standard deviation of the relevance score at each position.
    """
    # List to store the scores at each position for each epoch
    Scores_pos = []
    f1_scores = []
    
    # Move inputs and labels to the specified device
    train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)


    # Create a neural network model with the current hyperparameters
    model = ModelClassification(
        input_dim=train_inputs.shape[1],
        output_dim=output_dim,
        layer_dim=hyperparam_dict['layer_dim'],
        number_hidden_layer=hyperparam_dict['number_hidden_layer'],
        dropout_prob=hyperparam_dict['dropout_prob'],
        l2_regu=hyperparam_dict['l2_regu']
    ).to(device)


    # Set hyperparameters for optimization
    num_epochs = hyperparam_dict["num_epochs"]
    w_decay = hyperparam_dict['weight_decay']
    learning_rate = hyperparam_dict['learning_rate']
    batch_size = hyperparam_dict['batch_size']

    # Initialize Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

        
    for i in range(nbr_training):
        #training loop
        for epoch in range(num_epochs):
            model.train()
            #iterate on all the data of the current batch
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Zero the gradients, forward pass, backward pass, and optimization
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = model.get_loss(outputs, labels)
                loss.backward()
                optimizer.step()          
        # Generate predictions for the validation set
        val_pred = predict(model, val_inputs, device)
        # Compute the weighted F1 score for the validation set
        val_f1 = f1_score(val_labels, val_pred, average='weighted')
        # compute the relevance score for each position
        score = feature_importances_neural(model, columns_info, smoothness=0, pos=pos, plot=0)
        
        f1_scores.append(val_f1)
        Scores_pos.append(score)

    # Calculate mean and standard deviation of F1 scores and relevance score across all training
    return np.mean(f1_scores), np.std(f1_scores), np.mean(Scores_pos,axis=0), np.std(Scores_pos, axis=0)


def optimize_hyperparameters_nn_consistency(train_inputs, train_labels, val_inputs, val_labels,output_dim,columns_info, param_grid, nbr_training = 10, pos = range(0,600)):
    """Optimize hyperparameters for a neural network using batched cross-validation and the standard deviation of the relevance score as a criteria.

    Args:
        train_inputs (numpy array): Training data.
        train_labels (numpy array): True labels for training data.
        val_inputs (numpy array): Validation data.
        val_labels (numpy array): True labels for validation data.
        output_dim (int): Dimension of the output (number of classes).
        columns_data (pd.DataFrame): columns informations of the dataset.
        param_grid (dict): Dictionary of hyperparameter values to search.
        nbr_training (int): Number of times the neural net will be retrained to compute de std of the scores.
        pos (list of int): list of the position on wich to learn

    Returns:
        tuple of (dict, float, float): Best hyperparameters, best mean F1 score, best F1 score standard deviation.
    """
    # Calculate the batch size based on the specified number of batches
    best_params = None
    best_mean_f1_score = 0.0
    best_std_f1_score = 0.0
    best_mean_scores = 10.0
    best_std_scores = 10.0
    
    store_params = []
    store_f1 = []
    store_f1_std = []
    store_mean_scores = []
    store_std_scores = []
    
    start_time = time.time()

    # Generate all combinations of hyperparameters
    all_hyperparams = list(itertools.product(*param_grid.values()))

    # Iterate through hyperparameter combinations
    for hyperparams in all_hyperparams:
        # Create a dictionary mapping hyperparameter names to values
        hyperparam_dict = dict(zip(param_grid.keys(), hyperparams))
        
        # Use GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Perform cross-validation using batched training
        mean_f1, std_f1, mean_scores, std_scores = cross_val_pos_consistency(dict(zip(param_grid.keys(), hyperparams)), train_inputs,train_labels, val_inputs, val_labels,output_dim,columns_info,device,nbr_training, pos)

        # Print the results for the current hyperparameter combination
        print(f"time: {time.time()-start_time}\n")
        print(f"Hyperparameters: {hyperparam_dict}, Mean Validation F1 Score: {mean_f1}, Std Validation F1 Score: {std_f1}, Mean Std of Scores: {np.mean(std_scores)}\n")

        
        # store the relevant metrics
        store_params.append(hyperparam_dict)
        store_f1.append(mean_f1)
        store_f1_std.append(std_f1)
        store_mean_scores.append(mean_scores)
        store_std_scores.append(np.mean(std_scores))

        # Update the best hyperparameters if the mean score improves
        if np.mean(std_scores) < np.mean(best_std_scores):
            best_mean_scores = mean_scores
            best_std_scores = std_scores
            
            best_mean_f1_score = mean_f1
            best_std_f1_score = std_f1
            best_params = hyperparam_dict

    # Calculate total runtime
    end_time = time.time()
    runtime = end_time - start_time

    # Print the best hyperparameters and associated metrics
    print("\nBest Hyperparameters:")
    print(best_params)
    print(f"Best Mean Validation F1 Score: {best_mean_f1_score}")
    print(f"Std Validation F1 Score: {best_std_f1_score}")
    print(f"Best Mean Std of scores: {np.mean(best_std_scores)}")
    print(f"\n stored f1 scores: {store_f1}")
    print(f"\n stored consistency: {store_std_scores}")
    print(f"\n Total Runtime: {runtime} seconds")

    return best_params, best_mean_scores, best_std_scores