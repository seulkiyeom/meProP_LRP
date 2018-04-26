import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Conv2d(nn.Module):
   def __init__(self, input_dim, output_dim, kernel_size):
       super(Conv2d, self).__init__()
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.kernel_size = kernel_size
       self.layer = nn.Conv2d(self.input_dim, self.output_dim, self.kernel_size)


   def forward(self, input_tensor):
       return self.layer.forward(input_tensor)

   def lrp(self, R, lrp_var=None,param=None):
       print(self.kernel_size)

       return 0