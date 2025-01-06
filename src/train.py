import torch
import torch.nn as nn
import torch.optim as optim
from data_preprocessing import get_data_loaders
from model import ResNet18

def train(num_epochs=10, batch_size=64, learning_rate=0.001):
    """Train the ResNet model on the MNIST dataset."""
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Initialize model, loss function, and optimizer
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Print average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

        # Evaluate the model on the test set
        evaluate(model, test_loader, device)

def evaluate(model, test_loader, device):
    """Evaluate the model on the test dataset."""
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on the test set: {accuracy:.2f}%')

if __name__ == "__main__":
    train(num_epochs=10, batch_size=64, learning_rate=0.001)
