import torch.utils.data

def DataLoader(Dataset):
    return torch.utils.data.DataLoader(Dataset, batch_size= 4, shuffle= True)