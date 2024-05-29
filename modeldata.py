import torch
from torch.utils.data import  Dataset



class ModelData(Dataset):
    def __init__(self, text_data, price_data, labels):
        self.text_data = text_data
        self.price_data = price_data
        self.labels = labels

        self.text_data = torch.from_numpy(text_data)
        self.price_data = torch.from_numpy(price_data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text, prices, labels = self.text_data[idx], self.price_data[idx], self.labels[idx]
        labels = labels.float()
        
        return text, prices, labels
