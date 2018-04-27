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
       # image_patches = self.extract_patches()
       # Z = self.compute_z(image_patches)
       # Zs = self.compute_zs(Z)
       # result = self.compute_result(Z, Zs)

       hpool = wpool = self.layer.kernel_size
       hstride = wstride = self.layer.stride

       Hout = int((self.in_h - hpool) / hstride + 1)
       Wout = int((self.in_w - wpool) / wstride + 1)

       Rx = torch.zeros_like(self.input).float()

       for i in range(Hout):
           for j in range(Wout):
               Z = self.output[:, :, i:i + 1, j:j + 1] == self.input[:, :, i * hstride:i * hstride + hpool, j * wstride:j * wstride + wpool]
               Zs = (torch.sum(torch.sum(Z, dim=2, keepdim=True),dim=3, keepdim=True)).float()
               Zs += 1e-12 * (where(Zs >= 0, torch.ones_like(Zs), torch.ones_like(Zs) * -1))

               Rx[:, :, i * hstride:i * hstride + hpool, j * wstride:j * wstride + wpool] += torch.div(Z.float(), Zs) * self.R[:, :, i:i + 1, j:j + 1]
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

   def extract_patches(self):
       return torch.reshape(self.input, [self.in_N, self.Hout, self.Wout, self.layer.kernel_size[0], self.layer.kernel_size[1], self.in_depth])

   def compute_z(self, image_patches):
       Z = torch.eq(torch.reshape(self.output, [self.in_N, self.Hout, self.Wout, 1, 1, self.in_depth,]), image_patches)
       return where(Z, torch.ones_like(Z).float(), torch.zeros_like(Z).float())

   def compute_zs(self, Z, stabilizer=True, epsilon=1e-12):
       Zs = tf.reduce_sum(Z, [2, 3, 4], keep_dims=True)  # + tf.expand_dims(self.biases, 0)
       if stabilizer == True:
           stabilizer = epsilon * (tf.where(tf.greater_equal(Zs, 0), tf.ones_like(Zs, dtype=tf.float32),
                                            tf.ones_like(Zs, dtype=tf.float32) * -1))
           Zs += stabilizer
       return Zs