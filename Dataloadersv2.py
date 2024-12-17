import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class ReadDataset(Dataset):
    def __init__(self, source):
        if isinstance(source, str):
            #loads data from the file path
            data = np.load(source, allow_pickle=True)
            self.data = torch.from_numpy(data).float()
        else:
            #assume source is a numpy array
            self.data = torch.from_numpy(source).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def RandomSplit(datasets, train_set_percentage):
    lengths = [int(len(datasets)*train_set_percentage), len(datasets)-int(len(datasets)*train_set_percentage)]
    return random_split(datasets, lengths)







def GetDataLoaders(train_source, test_source, batch_size, shuffle=True, num_workers=0, pin_memory=True):
    train_set = ReadDataset(train_source)
    test_set = ReadDataset(test_source)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader

