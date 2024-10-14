import torch
import torch.nn as nn

class DisasterTweetClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1):
        """
        Initializes the neural network for binary classification.

        :param input_size: int
            The size of the input features (BERT embeddings size)
        :param hidden_size: int
            The size of the hidden layers
        :param output_size: int
            The size of the output layer (1 for binary classification)
        """
        super().__init__()

        # Fully connected layers
        self.layer1 = nn.Linear(input_size, hidden_size * 4)
        self.layer2 = nn.Linear(hidden_size * 4, hidden_size * 4)
        self.layer3 = nn.Linear(hidden_size * 4, hidden_size * 3)
        self.layer4 = nn.Linear(hidden_size * 3, hidden_size * 2)
        self.layer5 = nn.Linear(hidden_size * 2, hidden_size)

        # Output layer
        self.output = nn.Linear(hidden_size, output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)

        # Batch Normalization layers
        self.batchnorm1 = nn.BatchNorm1d(hidden_size * 4)
        self.batchnorm2 = nn.BatchNorm1d(hidden_size * 4)
        self.batchnorm3 = nn.BatchNorm1d(hidden_size * 3)
        self.batchnorm4 = nn.BatchNorm1d(hidden_size * 2)
        self.batchnorm5 = nn.BatchNorm1d(hidden_size)

        # Activation function
        self.relu = nn.ReLU()

        # Sigmoid for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Tensor
            Input tensor of size (batch_size, input_size)
        :return: Tensor
            Output logits for each class
        """
        # Layer 1
        x = self.layer1(x)
        x = self.batchnorm1(x)  # Apply BatchNorm after the Linear layer
        x = self.relu(x)

        # Layer 2
        x = self.layer2(x)
        x = self.batchnorm2(x)  # Apply BatchNorm after the Linear layer
        x = self.relu(x)

        # Layer 3
        x = self.layer3(x)
        x = self.batchnorm3(x)  # Apply BatchNorm after the Linear layer
        x = self.relu(x)

        # Layer 4
        x = self.layer4(x)
        x = self.batchnorm4(x)  # Apply BatchNorm after the Linear layer
        x = self.relu(x)

        # Layer 5
        x = self.layer5(x)
        x = self.batchnorm5(x)  # Apply BatchNorm after the Linear layer
        x = self.relu(x)

        # Output layer (no batch norm here)
        x = self.output(x)
        x = self.sigmoid(x)  # Apply sigmoid for binary output

        return x
