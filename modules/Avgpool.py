import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class AvgPool2d(nn.Module):
   def __init__(self, kernel_size):
       super(AvgPool2d, self).__init__()
       self.kernel_size = kernel_size
       self.layer = nn.AvgPool2d(self.kernel_size)

   def forward(self, input_tensor):
       return self.layer.forward(input_tensor)
   #
   def lrp(self, R, lrp_var=None,param=None):
       print(self.input)

       return 0