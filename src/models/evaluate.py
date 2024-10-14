import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix


def evaluate_model(model, X_test, y_test):
    # Convert test data to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)  # BCELoss expects float labels

    # Get predictions from the model
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        print(outputs)

        # Binary predictions (0 or 1) based on the sigmoid output
        predicted = (outputs > 0.5).int()

    # Calculate accuracy
    accuracy = accuracy_score(y_test_tensor, predicted)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # Calculate F1 score
    f1 = f1_score(y_test_tensor, predicted)
    print(f'F1 Score: {f1 * 100:.2f}%')

    # Calculate recall
    recall = recall_score(y_test_tensor, predicted)
    print(f'Recall: {recall * 100:.2f}%')

    # Compute confusion matrix
    confusion = confusion_matrix(y_test_tensor, predicted)
    print(confusion)
