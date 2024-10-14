import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .model import DisasterTweetClassifier


def train_model(X_train, y_train, input_size, hidden_size, num_epochs=200, batch_size=400, learning_rate=0.0003):
    # Try using GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Training is running on {device.upper()}.')

    # Convert data to PyTorch tensors
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # Initialize the model, loss function, and optimizer
    model = DisasterTweetClassifier(input_size, hidden_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0

        for i, (features, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {average_loss:.4f}')

    return model
