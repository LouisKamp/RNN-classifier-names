
import json
from typing import List, Tuple
from torch import Tensor
from torch.utils.data import Dataset
import torch as pt
from utils import char_to_tensor


class NamesDataset(Dataset):
    def __init__(self, train: bool = True):
        if (train):
            f = open('./data/processed/train_data.json', 'r')
        else:
            f = open('./data/processed/test_data.json', 'r')
        self.data_list: List[Tuple[str, int]] = json.loads(f.read())
        self.len = len(self.data_list)

    def __getitem__(self, idx):
        name, label = self.data_list[idx]
        name_char_tensors = []
        for char in name:
            name_char_tensors.append(char_to_tensor(char))

        return name_char_tensors, label

    def __len__(self):
        return self.len
