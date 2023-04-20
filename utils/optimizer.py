import torch

def optimizer(parameters):
    return torch.optim.SGD(params = parameters, lr = 0.001)


