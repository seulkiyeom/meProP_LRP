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
import plotly.plotly as py
import plotly.graph_objs as go

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

def visualize_3d(relevances, images_tensor=None, image_show=False, image_save=False):
    n, w, h, dim = relevances.shape
    heatmaps = []
    count = 0
    #import pdb;pdb.set_trace()
    for h,heat in enumerate(relevances):
        if images_tensor is not None:
            input_image = images_tensor[h]
            maps, new_input = render.hm_to_rgb_3d(heat, input_image, scaling = 3, sigma = 2, cmap = 'jet')
        else:
            maps, new_input = render.hm_to_rgb_3d(heat, scaling = 3, sigma = 2, cmap = 'jet')

        gap = np.zeros((new_input.shape[0], 2, new_input.shape[2]))
        image = np.hstack((new_input, gap, maps))

        if image_show:
            plt.imshow(image)
        if image_save:
            plt.imsave('saved_image/' + str(count), image, format="png")
            count += 1

def plot_relevances(rel, img, image_show = False, image_save = False):
    rel = rel.permute(0, 2, 3, 1).data.numpy()
    img = img.permute(0, 2, 3, 1).data.numpy()

    visualize(rel, img, image_show, image_save)

def plot_relevances_3d(rel, img, image_show = False, image_save = False):
    rel = rel.mean(axis=1, keepdims=True)
    rel = rel.transpose(0, 2, 3, 1)
    img = img.transpose(0, 2, 3, 1)

    visualize_3d(rel, img, image_show, image_save)