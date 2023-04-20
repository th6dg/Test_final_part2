import torch.nn as nn

def Loss_Fn(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)