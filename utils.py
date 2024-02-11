import torch
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import copy
import random
from tqdm import trange, tqdm
from multiprocessing import cpu_count

def cycle(dl):
    while True:
        for data in dl:
            yield data

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"make directory {path} successfully!")
    else:
        print(f"{path} already exits!")

def closest_factors(a):
    factors = []
    for i in range(1, int(a ** 0.5) + 1):
        if a % i == 0:
            factors.append((i, a // i))
    closest_factor = min(factors, key=lambda x: abs(x[0] - x[1]))
    return closest_factor

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))
        
# class CustomDataset(Dataset):  # ori
#     def __init__(self, data_path, labels_path):
#         self.data = np.load(data_path)
#         self.labels = np.load(labels_path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         data_item = self.data[index]
#         label_item = self.labels[index]
#         return data_item, label_item
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_item = self.data[index]
        return data_item