import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

def where(condition, x, y):
    return Variable(condition.float()) * x + Variable((condition != 1).float()) * y

class MaxPool2d(nn.Module):
   def __init__(self, kernel_size):
       super(MaxPool2d, self).__init__()
       self.kernel_size = kernel_size
       self.layer = nn.MaxPool2d(self.kernel_size)

   def forward(self, input_tensor):
       self.in_N, self.in_depth, self.in_h, self.in_w = input_tensor.size()
       return self.layer.forward(input_tensor)
   #
   def lrp(self, R, lrp_var=None,param=None):
       if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
           return self._simple_lrp(R)
       elif lrp_var.lower() == 'alphabeta' or lrp_var.lower() == 'alpha':
           return self._alphabeta_lrp(R, param)

   def _simple_lrp(self, R):
       self.check_shape(R)

       hpool = wpool = self.layer.kernel_size
       hstride = wstride = self.layer.stride

       Rx = torch.zeros(self.input.size())
       for i in range(self.Hout):
           for j in range(self.Wout):
               # Z = torch.eq(self.output[:, :, i:i+1, j:j + 1], self.input[:, :, i * hstride:i * hstride + hpool, j * wstride:j * wstride + wpool])
               # Z = where(Z, torch.ones_like(Z.float()), torch.zeros_like(Z.float()))
               Z = self.input[:, :, i * hstride:i * hstride + hpool, j * wstride:j * wstride + wpool]
               Zs = (torch.sum(torch.sum(Z, dim=2, keepdim=True),dim=3, keepdim=True))
               Zs += 1e-12 * where(Zs >= 0, torch.ones_like(Zs), torch.ones_like(Zs) * -1)

               Rx[:, :, i * hstride:i * hstride + hpool, j * wstride:j * wstride + wpool] += torch.div(Z, Zs) * self.R[:, :, i:i + 1, j:j + 1]
       return Rx

   def _alphabeta_lrp(self,R,alpha):
        return self._simple_lrp(R)

   def check_shape(self, R):
       self.R = R
       R_shape = self.R.size()
       output_shape = self.output.size()
       if len(R_shape) != 4:
           self.R = torch.reshape(self.R, output_shape)
       N, NF, self.Hout, self.Wout = self.R.size()