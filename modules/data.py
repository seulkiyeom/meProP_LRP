'''
Codes for loading the MNIST data
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import os
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

class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=True, transform=None, target_transform=None):

        self.root = os.path.expanduser(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        super(MyCustomDataset, self).__init__()

        if self.train:
            self.train_labels = []
            self.train_data = numpy.load(data_path)['arr_0.npy']
            self.train_labels = numpy.load(data_path)['arr_1.npy']
            # self.train_labels = convert_to_one_hot(self.train_labels, num_classes=2)
        else:
            self.test_labels = []
            self.test_data = numpy.load(data_path)['arr_0.npy']
            self.test_labels = numpy.load(data_path)['arr_1.npy']
            # self.test_labels = convert_to_one_hot(self.test_labels, num_classes=2)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

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
    Get CIFAR10 dataset
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

def get_medical_data(datapath='../data/data_HE_ts200/', download=True):
    '''
    Get Medical Data from Philipp

    Number of Data for microscopic image
        Train - 0 : 668
              - 1 : 668

              Tot : 1336

        Test  - 0 : 167
              - 1 : 167

              Tot : 334

        Total : 1670 (835 + 835)
    '''
    IMAGE_SIZE = 200
    IMAGE_DEPTH = 3
    NUM_CLASSES = 2
    NUM_TRAIN_SAMPLES = 1336
    NUM_TEST_SAMPLES = 334

    train_dataset = MyCustomDataset(data_path='../data/data_HE_ts200/train.npz', train=True,
                                   transform=transforms.ToTensor())
    test_dataset = MyCustomDataset(data_path='../data/data_HE_ts200/test.npz', train=False,
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


def convert_to_one_hot(Y, num_classes=None):
    Y = numpy.eye(num_classes)[Y.reshape(-1)]
    return Y.astype(int)