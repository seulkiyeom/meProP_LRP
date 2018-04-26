import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ReLU(nn.Module):
   def __init__(self, inplace=False):
       super(ReLU, self).__init__()
       self.inplace = inplace
       self.layer = nn.ReLU(inplace=self.inplace)

   def forward(self, input_tensor):
       return self.layer.forward(input_tensor)
   #
   def lrp(self, R, lrp_var=None,param=None):
       print(self.input)

       return 0