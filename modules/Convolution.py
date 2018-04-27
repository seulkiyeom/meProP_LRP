import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable

def where(condition, x, y):
    return Variable(condition.float()) * x + Variable((condition != 1).float()) * y

class Conv2d(nn.Module):
   def __init__(self, input_dim, output_dim, kernel_size):
       super(Conv2d, self).__init__()
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.kernel_size = kernel_size
       self.layer = nn.Conv2d(self.input_dim, self.output_dim, self.kernel_size)


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

       NF, df, hf, wf = self.layer.weight.shape
       hstride, wstride = self.layer.stride

       Rx = torch.zeros_like(self.input).float()

       for i in range(self.Hout):
           for j in range(self.Wout):
               Z = torch.unsqueeze(self.layer.weight, 0) * torch.unsqueeze(self.input[:, :, i * hstride:i * hstride + hf, j * wstride:j * wstride + wf], 1)
               Zs = Z.sum(2, keepdim=True).sum(3, keepdim=True).sum(4, keepdim=True) + torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.layer.bias, 0), 2), 2), 2)
               Zs += 1e-12 * (where(Zs >= 0, torch.ones_like(Zs), torch.ones_like(Zs) * -1))
               Rx[:, :, i * hstride:i * hstride + hf:, j * wstride:j * wstride + wf:] += ((Z / Zs) * torch.unsqueeze(self.R[:, :, i:i + 1, j:j + 1],-1)).sum(dim=1)
       return Rx

   def _alphabeta_lrp(self, R, alpha):
       beta = 1 - alpha
       self.check_shape(R)

       NF, df, hf, wf = self.layer.weight.shape
       hstride, wstride = self.layer.stride

       Rx = torch.zeros_like(self.input).float()

       for i in range(self.Hout):
           for j in range(self.Wout):
               Z = torch.unsqueeze(self.layer.weight, 0) * torch.unsqueeze(
                   self.input[:, :, i * hstride:i * hstride + hf, j * wstride:j * wstride + wf], 1)

               if not alpha == 0:
                   Zp = where(Z > 0, Z, torch.zeros_like(Z))
                   Bp = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(where(self.layer.bias > 0, self.layer.bias, torch.zeros_like(self.layer.bias)), 0), 2), 2), 2)
                   Zsp = Zp.sum(2, keepdim=True).sum(3, keepdim=True).sum(4, keepdim=True) + Bp
                   Ralpha = alpha * ((Zp/Zsp) * torch.unsqueeze(self.R[:, :, i:i + 1, j:j + 1], -1)).sum(dim=1)
               else:
                   Ralpha = 0

               if not beta == 0:
                   Zn = where(Z < 0, Z, torch.zeros_like(Z))
                   Bn = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(
                       torch.unsqueeze(where(self.layer.bias < 0, self.layer.bias, torch.zeros_like(self.layer.bias)),
                                       0), 2), 2), 2)
                   Zsn = Zn.sum(2, keepdim=True).sum(3, keepdim=True).sum(4, keepdim=True) + Bn
                   Rbeta = beta * ((Zn / Zsn) * torch.unsqueeze(self.R[:, :, i:i + 1, j:j + 1], -1)).sum(dim=1)
               else:
                   Rbeta = 0

               Rx[:, :, i * hstride:i * hstride + hf:, j * wstride:j * wstride + wf:] += Ralpha + Rbeta

       return Rx

               # Zs = Z.sum(axis=(1, 2, 3), keepdims=True) + self.B[na, na, na, na, ...]
               # Zs += 1e-12 * ((Zs >= 0) * 2 - 1.)  # add a weak numerical stabilizer to cushion division by zero
               # Rx[:, i * hstride:i * hstride + hf:, j * wstride:j * wstride + wf:, :] += (
               #             (Z / Zs) * R[:, i:i + 1, j:j + 1, na, :]).sum(axis=4)
               #
               #
               #
               #
               #
               # if not alpha == 0:
               #
               #     Bp = (self.B * (self.B > 0))[na, na, na, na, ...]
               #     Zsp = Zp.sum(axis=(1, 2, 3), keepdims=True) + Bp
               #     Ralpha = alpha * ((Zp / Zsp) * R[:, i:i + 1, j:j + 1, na, :]).sum(axis=4)
               # else:
               #     Ralpha = 0
               #
               # if not beta == 0:
               #
               #     Bn = (self.B * (self.B < 0))[na, na, na, na, ...]
               #     Zsn = Zn.sum(axis=(1, 2, 3), keepdims=True) + Bn
               #     Rbeta = beta * ((Zn / Zsn) * R[:, i:i + 1, j:j + 1, na, :]).sum(axis=4)
               # else:
               #     Rbeta = 0





   def check_shape(self, R):
       self.R = R
       R_shape = self.R.size()
       output_shape = self.output.size()
       if len(R_shape) != 4:
           self.R = torch.reshape(self.R, output_shape)
       N, NF, self.Hout, self.Wout = self.R.size()


