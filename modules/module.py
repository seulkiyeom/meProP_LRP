import numpy
import torch

def LRP(activations, R, lrp_var, param):
    check_shape(activations, R)


    return R

def check_shape(activations, R):
    R_shape = list(R.data.size())
    activations_shape = list(activations.data.size())
