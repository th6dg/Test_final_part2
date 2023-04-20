import matplotlib.pyplot as plt
import torch

def plot_from_tensor(image_tensor):
    image_array = image_tensor.numpy()
    plt. imshow(image_tensor)
    plt.show()