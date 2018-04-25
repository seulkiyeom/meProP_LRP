import torch
import torch.nn as nn
import torch.nn.functional as F

def save_model(net, optim, ckpt_fname):
    state_dict = net.module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optim},
        ckpt_fname)