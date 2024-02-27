import itertools
import time
import numpy as np
from neuralnet import ModelClassification
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

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

def cross_val_score(hyperparam_dict, train_loader, val_inputs, val_labels, device):
    """Perform cross-validation for a neural network model.

    Args:
        hyperparam_dict (dict): Dictionary mapping hyperparameter names to values.
        train_loader (DataLoader): DataLoader for training data.
        val_inputs (numpy array): Validation data.
        val_labels (numpy array): True labels for validation data.
        device (torch.device): The device (cuda or cpu) on which the model should run.

    Returns:
        tuple of floats: Mean and standard deviation of F1 scores for all epochs.
    """
    # List to store F1 scores for each epoch
    f1_scores = []
    

    # Iterate through batches in the DataLoader
    for inputs, labels in train_loader:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        # Create a neural network model with the current hyperparameters
        model = ModelClassification(
            input_dim=inputs.shape[1],
            output_dim=4,
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

        # Create DataLoader for training data in the current batch
        sub_train_dataset = torch.utils.data.TensorDataset(inputs, labels)
        sub_train_loader = torch.utils.data.DataLoader(dataset=sub_train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

        #training loop
        for epoch in range(num_epochs):
            model.train()
            #iterate on all the data of the current batch
            for sub_inputs, sub_labels in sub_train_loader:
                sub_inputs, sub_labels = sub_inputs.to(device), sub_labels.to(device)

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
        f1_scores.append(val_f1)

    # Calculate mean and standard deviation of F1 scores across all epochs
    return np.mean(f1_scores), np.std(f1_scores)


def optimize_hyperparameters_nn(train_inputs, train_labels, val_inputs, val_labels, number_batch, param_grid):
    """Optimize hyperparameters for a neural network using batched cross-validation.

    Args:
        train_inputs (numpy array): Training data.
        train_labels (numpy array): True labels for training data.
        val_inputs (numpy array): Validation data.
        val_labels (numpy array): True labels for validation data.
        num_epochs (int): Number of training epochs per batch.
        number_batch (int): Number of batches for cross-validation.
        param_grid (dict): Dictionary of hyperparameter values to search.

    Returns:
        tuple of (dict, float, float): Best hyperparameters, best mean F1 score, best F1 score standard deviation.
    """
    # Calculate the batch size based on the specified number of batches
    batch_size = int(train_inputs.shape[0] / number_batch)
    best_params = None
    best_mean_f1_score = 0.0
    best_std_f1_score = 0.0
    start_time = time.time()

    # Generate all combinations of hyperparameters
    all_hyperparams = list(itertools.product(*param_grid.values()))

    # Iterate through hyperparameter combinations
    for hyperparams in all_hyperparams:
        # Create a dictionary mapping hyperparameter names to values
        hyperparam_dict = dict(zip(param_grid.keys(), hyperparams))
        
        # Use GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # Create DataLoader for training data
        train_dataset = TensorDataset(torch.Tensor(train_inputs), torch.LongTensor(train_labels))
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

        # Perform cross-validation using batched training
        mean_f1, std_f1 = cross_val_score(dict(zip(param_grid.keys(), hyperparams)), train_loader, val_inputs, val_labels,device)

        # Print the results for the current hyperparameter combination
        print(f"time: {time.time()-start_time}\n")
        print(f"Hyperparameters: {hyperparam_dict}, Mean Validation F1 Score: {mean_f1}, Std Validation F1 Score: {std_f1}\n")

        # Update the best hyperparameters if the mean F1 score improves
        if mean_f1 > best_mean_f1_score:
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
    print(f"Total Runtime: {runtime} seconds")

    return best_params, best_mean_f1_score, best_std_f1_score