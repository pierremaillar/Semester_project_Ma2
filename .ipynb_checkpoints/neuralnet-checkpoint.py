import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class ModelClassification(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dim, number_hidden_layer, dropout_prob=0.0, l2_regu=0.0):
        """Initialize a neural network model for classification.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of the output (number of classes).
            layer_dim (int): Dimension of hidden layers.
            number_hidden_layer (int): Number of hidden layers.
            dropout_prob (float): Dropout probability for regularization (default is 0.0).
            l2_regu (float): L2 regularization strength (default is 0.0).
        """
        super(ModelClassification, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_dim, layer_dim)
        self.input_phi = nn.Softplus()
        self.dropout = nn.Dropout(dropout_prob)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(number_hidden_layer - 1):
            self.hidden_layers.append(nn.Linear(layer_dim, layer_dim))
            self.hidden_layers.append(nn.Softplus())

        # Output layer
        self.output_layer = nn.Linear(layer_dim, output_dim)

        self.l2_reg = l2_regu

    def forward(self, x):
        """Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output predictions.
        """
        x = self.input_layer(x)
        x = self.input_phi(x)
        x = self.dropout(x)

        for i in range(0, len(self.hidden_layers), 2):
            x = self.hidden_layers[i](x)
            x = self.hidden_layers[i + 1](x)

        x = self.output_layer(x)

        return x

    def get_loss(self, output, target):
        """Compute the loss function for training.

        Args:
            output (torch.Tensor): Model predictions.
            target (torch.Tensor): True labels.

        Returns:
            torch.Tensor: Total loss including cross-entropy and L2 regularization.
        """
        cross_entropy_loss = nn.CrossEntropyLoss()(output, target)

        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
        loss = cross_entropy_loss + 0.5 * self.l2_reg * l2_reg

        return loss

def train_model(model, num_epochs, train_inputs, train_labels, val_inputs, val_labels, optimizer, batch_size=128):
    """Train a neural network model.

    Args:
        model (ModelClassification): The neural network model.
        num_epochs (int): Number of training epochs.
        train_inputs (torch.Tensor): Training input data.
        train_labels (torch.Tensor): True labels for training data.
        val_inputs (torch.Tensor): Validation input data.
        val_labels (torch.Tensor): True labels for validation data.
        optimizer (torch.optim): Optimizer for training.
        batch_size (int): Batch size for training (default is 128).

    Returns:
        float: Validation accuracy after training.
    """
    # Use GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_inputs = train_inputs.to(device)
    train_labels = train_labels.to(device)
    val_inputs = val_inputs.to(device)
    val_labels = val_labels.to(device)

    # Create DataLoader for training data
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_labels)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients, forward pass, backward pass, and optimization
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.get_loss(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation on the validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_inputs)
            val_loss = model.get_loss(val_outputs, val_labels)
            val_pred = val_outputs.argmax(dim=1)
            val_correct = (val_pred == val_labels).sum().item()
            val_accuracy = val_correct / val_inputs.shape[0]

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100*val_accuracy:.4f}%")

    return val_accuracy

def features_importances_nn(model_neural, data_df, smoothness=30, nbr_pos=599, plot=0):
    """Calculate feature importances for a neural network model.

    Args:
        model_neural (ModelClassification): Trained neural network model.
        data_df (pd.DataFrame): Dataframe containing feature names.
        smoothness (int): Size of the smoothing window (default is 30).
        nbr_pos (int): Number of positions (default is 599).
        plot (int): Flag for plotting the feature importance graph (default is 0).

    Returns:
        pd.DataFrame: Smoothed feature importances.
    """
    # Extract feature names from the DataFrame
    columns_data = data_df.drop(data_df.columns[0], axis=1).columns
    scores = pd.DataFrame(index=[f'pos_{i}' for i in range(1, nbr_pos + 1)])

    # Compute neural network feature importances
    #computing the features importance in the neural network
    eye = torch.tensor(np.eye(len(columns_data)), dtype=torch.float32)
    zero = torch.tensor(np.zeros(len(columns_data)), dtype=torch.float32)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        eye = eye.to(device)
        zero = zero.to(device)
    
    neural_impo = model_neural(eye)-model_neural(zero)
    
    if torch.cuda.is_available():
        neural_impo = neural_impo.cpu()
        
    neural_impo = neural_impo.detach().numpy()
    neural_impo = neural_impo.reshape((len(neural_impo), 4))
    neural_impo = pd.DataFrame(neural_impo.T, columns=columns_data)
    


    
    # Compute scores for each class and position
    for j in [0, 1, 2, 3]:
        neural_impo_temp = neural_impo.iloc[j]
        for i in range(1, nbr_pos + 1):
            selected_columns = neural_impo_temp.filter(regex=f'_{i}_')
            scores.loc[f'pos_{i}', f'Neural_scores_class{j}'] = np.sum(np.abs(selected_columns), axis=0)

    # Standardize scores
    scaler = StandardScaler()
    scores_normalized = pd.DataFrame(scaler.fit_transform(scores), columns=scores.columns, index=scores.index)

    # Smooth the score plot with a moving average
    scores_smoothed = scores_normalized.rolling(window=smoothness, min_periods=1, center=False).mean()
    scores_smoothed = pd.DataFrame(scaler.fit_transform(scores_smoothed), columns=scores_smoothed.columns,
                                   index=scores_smoothed.index)

    # Plot the feature importance graph if specified
    if plot == 1:
        plt.plot(range(1, nbr_pos + 1), scores_smoothed['Neural_scores_class0'], label='class_0', color='blue')
        plt.plot(range(1, nbr_pos + 1), scores_smoothed['Neural_scores_class1'], label='class_1', color='green')
        plt.plot(range(1, nbr_pos + 1), scores_smoothed['Neural_scores_class2'], label='class_2', color='red')
        plt.plot(range(1, nbr_pos + 1), scores_smoothed['Neural_scores_class3'], label='class_3', color='black')

        plt.title('Position importances')
        plt.xlabel('Positions')
        plt.ylabel('Scores')
        plt.legend()

    return scores_smoothed