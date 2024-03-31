import torch.nn as nn
from torch.utils.data.dataset import Dataset

class TextDataset(Dataset):
    def __init__(self, data_dir):
        super(TextDataset, self).__init__()
        self.data_dir = data_dir
        self.ids = list()
        return
    
    def forward(self):
        return
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def num_classes(self):
        return 2

def collater():
    return None

def word2em():
    return
