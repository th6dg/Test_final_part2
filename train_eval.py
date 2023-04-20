from utils.dataset import Mnist_Train, Mnist_Test
from utils.dataloader import DataLoader
from model.DungNet import DungNet
from utils.optimizer import optimizer
from utils.loss import Loss_Fn
from tqdm import tqdm
import torch

# Define device to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = './data'

# Dataset
train_dataset = Mnist_Train()
test_dataset = Mnist_Test()

# Data loader
train_loader = DataLoader(train_dataset)
test_loader = DataLoader(test_dataset)

# Init model, optimizer
model = DungNet()
optimizer = optimizer(model.parameters())


# Training loop
epochs = 3


for epoch in range (epochs):
    model.train()
    train_loss = 0
    
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # Optimizer with no gradients
        optimizer.zero_grad()
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = Loss_Fn(outputs, labels)
        # Backward pass 
        loss.backward()
        # Update parameters
        optimizer.step()
        train_loss += loss.item()
    print(f'Epoch: {epoch}, Loss value: {train_loss/15000}')
    
    # Set the model to evaluation mode
    model.eval()

    # Compute the accuracy on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move the data to the device
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the predicted labels
            _, predicted = torch.max(outputs.data, 1)

            # Update the counters
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the epoch number and test accuracy
    print(f"Epoch: {epoch}, Test Accuracy: {100*correct/total:.2f}%")
    # Save model
    torch.save(model.state_dict(), f"model{epoch}.pt")
    

