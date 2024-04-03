import torch.nn as nn
import re

from torch.utils.data.dataset import Dataset

from utils import tokenizer, read_file

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
        sentence = sentence.split("\t")
        return tokenizer(sentence[1]), sentence[0]
    
    def num_classes(self):
        return 2

def collater():
    return None
