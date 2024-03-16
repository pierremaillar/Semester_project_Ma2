import itertools
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def compute_batch_f1(model, inputs, labels):
    """Compute the weighted F1 score for a batch.
    Args:
        model (RandomForestClassifier): The trained Random Forest classifier.
        inputs (numpy array): Input data for predictions.
        labels (numpy array): True labels corresponding to the input data.
    Returns:
        float: Weighted F1 score for the batch.
    """
    # Make predictions using the trained model
    predictions = model.predict(inputs)
    
        # Compute the weighted F1 score
    f1 = f1_score(labels, predictions, average='weighted')
    return f1

def cross_val_score_batches(model, train_inputs, train_labels, val_inputs, val_labels, batch_size):
    """Perform cross-validation using batched training.
    Args:
        model (RandomForestClassifier): The Random Forest classifier.
        train_inputs (numpy array): Training data.
        train_labels (numpy array): True labels for training data.
        val_inputs (numpy array): Validation data.
        val_labels (numpy array): True labels for validation data.
        batch_size (int): Size of each batch.
    Returns:
        tuple of floats: Mean and standard deviation of F1 scores for all batches.
    """
    # Calculate the number of batches based on the training data
    num_batches = len(train_inputs) // batch_size
    batch_f1_scores = []

    # Create an array of indices for shuffling during batch training
    indices = np.arange(len(train_inputs))

    # Iterate through batches
    for i in range(num_batches):
        # Shuffle indices for each batch
        np.random.shuffle(indices)

        # Calculate start and end indices for the current batch
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size

        # Extract batch training inputs and labels using shuffled indices
        batch_train_inputs = train_inputs[indices[batch_start:batch_end]]
        batch_train_labels = train_labels[indices[batch_start:batch_end]]

        # Train the model on the current batch
        model.fit(batch_train_inputs, batch_train_labels)

        # Compute F1 score for the validation set using the current model
        f1 = compute_batch_f1(model, val_inputs, val_labels)
        batch_f1_scores.append(f1)

    # Calculate mean and standard deviation of F1 scores across all batches
    mean_f1 = np.mean(batch_f1_scores)
    std_f1 = np.std(batch_f1_scores)

    return float(mean_f1), float(std_f1)

def optimize_hyperparameters_rf(train_inputs, train_labels, val_inputs, val_labels, param_grid, number_batch):
    """Optimize Random Forest hyperparameters using batched cross-validation.
    Args:
        train_inputs (numpy array): Training data.
        train_labels (numpy array): True labels for training data.
        val_inputs (numpy array): Validation data.
        val_labels (numpy array): True labels for validation data.
        param_grid (dict): Dictionary of hyperparameter values to search.
        number_batch (int): Number of batches for cross-validation.
    Returns:
        tuple of (dict, float, float): Best hyperparameters, best mean F1 score, best F1 score standard deviation.
    """
    # Calculate the batch size based on the specified number of batches
    batch_size = int(train_inputs.shape[0] / number_batch)
    best_params = None
    best_mean_f1_score = 0.0
    best_std_f1_score = 0.0
    start_time = time.time()

    # Generate all possible combinations of hyperparameter values
    all_hyperparams = list(itertools.product(*param_grid.values()))

    # Iterate through hyperparameter combinations
    for hyperparams in all_hyperparams:
        # Create a dictionary mapping hyperparameter names to values
        hyperparam_dict = dict(zip(param_grid.keys(), hyperparams))

        # Create a Random Forest model with the current hyperparameters
        model = RandomForestClassifier(
            n_estimators=hyperparam_dict['n_estimators'],
            max_depth=hyperparam_dict['max_depth'],
            max_features=hyperparam_dict['max_features'],
            bootstrap=hyperparam_dict['bootstrap'],
            class_weight=hyperparam_dict['class_weight'],
            min_samples_leaf=hyperparam_dict['min_samples_leaf']
        )

        # Perform cross-validation using batched training
        mean_f1, std_f1 = cross_val_score_batches(model, train_inputs, train_labels, val_inputs, val_labels, batch_size)

        # Print the results for the current hyperparameter combination
        print(f"Hyperparameters: {hyperparam_dict}, Mean Validation F1 Score: {mean_f1}, Std Validation F1 Score: {std_f1}")

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

    return best_params, float(best_mean_f1_score), float(best_std_f1_score)