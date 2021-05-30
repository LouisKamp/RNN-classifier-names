
# %%
from typing import Dict, List
import numpy as np
import json


# %%

# import names

boy_names = np.loadtxt('./data/raw/alle-godkendte-drengenavne.csv', dtype=str, delimiter=',',
                       skiprows=1)
girl_names = np.loadtxt(
    './data/raw/alle-godkendte-pigenavne.csv', delimiter=',', dtype=str, skiprows=1)


all_names = np.append(boy_names, girl_names)


# %% calc dict

chars: Dict[str, int] = {}
index = 0

for name in all_names:
    for letter in name:
        if (letter).lower() not in chars:
            chars[(letter).lower()] = index
            index += 1

# %%
f = open('./data/processed/dict.json', 'w')
f.write(json.dumps(chars))
f.close()

# %%
