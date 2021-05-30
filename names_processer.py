# %%
from typing import Dict, List, Tuple
from utils import LABELS, get_dict_len, get_index
import torch as pt
import numpy as np
import json

# %%
boy_names = np.loadtxt('./data/raw/alle-godkendte-drengenavne.csv', dtype=str, delimiter=',',
                       skiprows=1)
girl_names = np.loadtxt(
    './data/raw/alle-godkendte-pigenavne.csv', delimiter=',', dtype=str, skiprows=1)


data: List[Tuple[str, int]] = []

# %%
for boy_name in boy_names:
    data.append((boy_name.lower(), LABELS['BOY']))

for girl_name in girl_names:
    data.append((girl_name.lower(), LABELS['GIRL']))

# %%
train_data = data[0::2]
f = open('./data/processed/train_data.json', 'w')
f.write(json.dumps(train_data))
f.close()

# %%
test_data = data[1::2]
f = open('./data/processed/test_data.json', 'w')
f.write(json.dumps(test_data))
f.close()

# %%
