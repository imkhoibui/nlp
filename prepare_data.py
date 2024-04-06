import torch.nn as nn
import re

from torch.utils.data.dataset import Dataset

from utils import read_file, split_line

class TextDataset(Dataset):
    def __init__(self, data_file, transform: None):
        super(TextDataset, self).__init__()
        self.data_file = read_file(data_file)
        self.transform = transform
        return
    
    def load_sentence(self, index: int):
        return self.data_file[index]
    
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, index):
        sentence = self.data_file[index]
        feature, target = split_line(sentence)
        return feature, target
    
    def num_classes(self):
        return 2

def collater():
    return None
