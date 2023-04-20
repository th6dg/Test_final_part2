from model.DungNet import DungNet
import torch
from utils.dataset import Dataset
from utils.dataloader import DataLoader
from utils.plot import plot_from_tensor
import matplotlib.pyplot as plt

# Init test set
test_dataset = Dataset('./data/mnist_test.csv')
test_loader = DataLoader(test_dataset)

# Init model
model = DungNet()
model.load_state_dict(torch.load("model0.pt"))

# Define the device to use for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get image to predict in test_dataset
test_data = test_dataset.__getitem__(6)
test_image = test_data[0]

# Make predictions on new data
with torch.no_grad():
    test_image = test_image.unsqueeze_(0).to(device)
    print(test_image.shape)
    predictions = model(test_image)
    _, predicted = torch.max(predictions.data, 1)

# Visualize
plt.title(predicted)    
plot_from_tensor(test_image.squeeze().squeeze())
