'''
Codes for loading the MNIST data
'''

import numpy
import torch
from torchvision import datasets, transforms


class PartDataset(torch.utils.data.Dataset):
    '''
    Partial Dataset:
        Extract the examples from the given dataset,
        starting from the offset.
        Stop if reach the length.
    '''

    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        super(PartDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.dataset[i + self.offset]


def get_mnist(datapath='./data/', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    # MNIST Dataset
    train_dataset = datasets.MNIST(root=datapath,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=download)

    test_dataset = datasets.MNIST(root=datapath,
                                  train=False,
                                  transform=transforms.ToTensor())
    return train_dataset, test_dataset

def get_cifar10(datapath='./data/', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    # MNIST Dataset
    train_dataset = datasets.CIFAR10(root=datapath,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=download)

    test_dataset = datasets.CIFAR10(root=datapath,
                                  train=False,
                                  transform=transforms.ToTensor())
    return train_dataset, test_dataset


def get_artificial_dataset(nsample, ninfeature, noutfeature):
    '''
    Generate a synthetic dataset.
    '''
    data = torch.randn(nsample, ninfeature).cuda()
    target = torch.LongTensor(
        numpy.random.randint(noutfeature, size=(nsample, 1))).cuda()
    return torch.utils.data.TensorDataset(data, target)
