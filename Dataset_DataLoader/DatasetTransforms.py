import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

dataset = torchvision.datasets.MNIST(
    root='./data', download=True, transform=torchvision.transforms.ToTensor()
)

class WineDataset(Dataset):
    def __init__(self, transform=None) -> None:
        super().__init__()
        # data loading 
        xy = np.loadtxt('../data/wine.csv', delimiter=",", dtype=np.float32, skiprows=1)
        # we do not convert to tensor here
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]
        self.transform = transform

    def __getitem__(self, index):
        sample =  self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def __len__(self):
        return self.n_samples
    

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self, factor) -> None:
        self.factor = factor
    
    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target

dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(2)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features)
print(type(features), type(labels))
