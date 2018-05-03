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
import modules.render as render
import numpy as np
import matplotlib.pyplot as plt

from modules.model import Net


def visualize(relevances, images_tensor=None, image_show=False, image_save=False):
    n, w, h, dim = relevances.shape
    heatmaps = []
    count = 0
    #import pdb;pdb.set_trace()
    if images_tensor is not None:
        assert relevances.shape==images_tensor.shape, 'Relevances shape != Images shape'
    for h,heat in enumerate(relevances):
        if images_tensor is not None:
            input_image = images_tensor[h]
            maps = render.hm_to_rgb(heat, input_image, scaling = 3, sigma = 2)
        else:
            maps = render.hm_to_rgb(heat, scaling = 3, sigma = 2)
        heatmaps.append(maps)
        if image_show:
            plt.imshow(maps)
        if image_save:
            plt.imsave('saved_image/' + str(count), maps, format="png")
            count += 1

    R = np.array(heatmaps)

def plot_relevances(rel, img, image_show = False, image_save = False):
    rel = rel.permute(0, 2, 3, 1).data.numpy()
    img = img.permute(0, 2, 3, 1).data.numpy()

    visualize(rel, img, image_show, image_save)




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

