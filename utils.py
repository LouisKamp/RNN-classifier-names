import json
from typing import Dict, List
import numpy as np
import torch as pt
from torch.functional import Tensor


f = open('./data/processed/dict.json', 'r')
chars: Dict[str, int] = json.loads(f.read())

LABELS: Dict[str, int] = {'BOY': 0, 'GIRL': 1}
LABELS_TO_TEXT: Dict[str, int] = {0: "BOY", 1: "GIRL"}


char_to_tensor_dict: Dict[str, Tensor] = {}

chars_len = len(chars)

for idx, char in enumerate(chars):
    char_tensor = pt.zeros(chars_len)
    char_tensor[idx] = 1
    char_to_tensor_dict[char] = char_tensor


def get_index(letter: str):
    return chars[letter]


def char_to_tensor(letter: str):
    return char_to_tensor_dict[letter]


def get_dict_len():
    return len(chars)
