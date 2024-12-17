import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
import math
from torch.utils.data import TensorDataset


def generate_random_function():
   #Generates a random mathematical function for RGB computation.
    functions = [
        lambda x, y, z: (x + y + z) % 1.0,  # Linear combination
        lambda x, y, z: (math.sin(x) + math.cos(y) + z) % 1.0,  # Trigonometric
        lambda x, y, z: (x * y * z) % 1.0,  # Multiplicative
        lambda x, y, z: ((x**2 + y**2 + z**2)**0.5) % 1.0  # Distance from origin
    ]
    return random.choice(functions)

class ReadDataset(Dataset):
    def __init__(self, source):
        if isinstance(source, str):
            data = np.load(source, allow_pickle=True)
            self.data = torch.from_numpy(data).float()
        else:
            self.data = torch.from_numpy(source).float()

        #Assign random functions for R, G, B per point cloud
        self.rgb_functions = [
            (generate_random_function(), generate_random_function(), generate_random_function())
            for _ in range(len(self.data))
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point_cloud = self.data[index]
    
        # Ensure only spatial coordinates are used for RGB calculation
        if point_cloud.shape[1] > 3:
            point_cloud = point_cloud[:, :3]  # Use only x, y, z
    
        r_func, g_func, b_func = self.rgb_functions[index]

        # Compute RGB values
        rgb_values = torch.tensor([[r_func(x, y, z), g_func(x, y, z), b_func(x, y, z)] for x, y, z in point_cloud])

    # Combine spatial and RGB values
        combined = torch.cat((point_cloud, rgb_values), dim=1)
        return combined


def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets) * train_set_percentage), len(datasets) - int(len(datasets) * train_set_percentage)]
    return random_split(datasets, lengths)

def GetDataLoaders(train_set, test_set, batch_size, shuffle=True, num_workers=0, pin_memory=True):
    #train_set = ReadDataset(train_source)
    #test_set = ReadDataset(test_source)
    train_set = TensorDataset(train_set)
    test_set = TensorDataset(test_set)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader
