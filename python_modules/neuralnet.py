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
        super().__init__()

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
        self.softmax = nn.Softmax(dim=1)

        self.l2_reg = l2_regu

    def forward(self, x):
        """Forward pass through the neural network.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output predictions.
        """
        x = self.input_layer(x.float())
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
        cross_entropy_loss = nn.CrossEntropyLoss()(output, target.long())

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
            #val_pred = torch.softmax(val_outputs, dim=1) #to give probabilities as output
            val_pred = val_outputs.argmax(dim=1)
            val_correct = (val_pred == val_labels).sum().item()
            val_accuracy = val_correct / val_inputs.shape[0]

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss.item():.4f}")
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {100*val_accuracy:.4f}%")

    return val_accuracy
