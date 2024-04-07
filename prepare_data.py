from json import load
from torch.utils.data.dataset import Dataset
import torch
import numpy as np

from utils import read_file, split_line, load_glove, word_to_vec

class TextDataset(Dataset):
    def __init__(self, data_file, transform: None):
        super(TextDataset, self).__init__()
        self.data_file = read_file(data_file)
        self.word_2_vec = load_glove()
        self.transform = transform
        return
    
    def load_sentence(self, index: int):
        return self.data_file[index]
    
    def __len__(self):
        return len(self.data_file)
    
    def __getitem__(self, index):
        sentence = self.data_file[index]
        features, target = split_line(sentence)
        target = float(target)
        features = np.array([word_to_vec(word, self.word_2_vec) for word in features])
        if self.transform:
            features = self.transform(features)
        return {"input" : torch.from_numpy(features), "category" : torch.Tensor([target])}
    
    def num_classes(self):
        return 2

def collater():
    return None
