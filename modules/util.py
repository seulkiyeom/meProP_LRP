# -*- coding: utf-8 -*-

'''
Helper class to facilitate experiments with different k
'''
import sys
import time
from statistics import mean

import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from modules.model import Net

class TestGroup(object):
    '''
    A network and k in meporp form a test group.
    Test groups differ in minibatch size, hidden features, layer number and dropout rate.
    '''

    def __init__(self,
                 args,
                 train_dataset,
                 batch_size,
                 test_batch_size,
                 hidden,
                 layer,
                 dropout,
                 kwargs,
                 test_dataset=None,
                 cudatensor=False,
                 file=sys.stdout):
        self.args = args
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.hidden = hidden
        self.layer = layer
        self.dropout = dropout
        self.file = file
        self.train_dataset = train_dataset


        self.trainloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True, **kwargs)
        if test_dataset:
            self.testloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                          batch_size=test_batch_size,
                                                          shuffle=False, **kwargs)
        else:
            self.testloader = None

        self.basettime = None
        self.basebtime = None

    def reset(self):
        '''
        Reinit the trainloader at the start of each run,
        so that the traning examples is in the same random order
        '''
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

    def run(self, k=None, epoch=None):
        '''
        Run a training loop.
        '''
        if k is None:
            k = self.args.k
        if epoch is None:
            epoch = self.args.n_epoch
        print(
            'mbsize: {}, hidden size: {}, layer: {}, dropout: {}, k: {}'.
            format(self.batch_size, self.hidden, self.layer, self.dropout, k),
            file=self.file)

        # Init the model, the optimizer and some structures for logging
        self.reset()

        # Build model
        model = Net(k, self.dropout)

