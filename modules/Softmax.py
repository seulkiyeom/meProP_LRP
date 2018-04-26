import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class LogSoftmax(nn.Module):
   def __init__(self, dim=None):
       super(LogSoftmax, self).__init__()
       self.dim = dim
       self.layer = nn.LogSoftmax(dim = self.dim)

   def forward(self, input_tensor):
       return self.layer.forward(input_tensor)

   def lrp(self, R, lrp_var=None,param=None):
       self.R = R

       Rx = self.input * self.R

       return Rx