import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

class Linear(nn.Module):
   def __init__(self, input_dim, output_dim):
       super(Linear, self).__init__()
       self.input_dim = input_dim
       self.output_dim = output_dim
       self.layer = nn.Linear(self.input_dim, self.output_dim)


   def forward(self, input_tensor):
       return self.layer.forward(input_tensor)
   #
   def lrp(self, R, lrp_var=None,param=None):
       if lrp_var is None or lrp_var.lower() == 'none' or lrp_var.lower() == 'simple':
           return self._simple_lrp(R)
       elif lrp_var.lower() == 'alphabeta' or lrp_var.lower() == 'alpha':
           return self._alphabeta_lrp(R, param)


   def _simple_lrp(self, R):
       self.R = R
       R_shape = self.R.size()
       if len(R_shape) != 2:
           output_shape = self.output.size()
           self.R = torch.reshape(self.R, output_shape)

       Z = torch.unsqueeze(torch.transpose(self.layer.weight, 0, 1), 0) * torch.unsqueeze(self.input, -1)
       Zs = torch.unsqueeze(torch.sum(Z, dim=1), 1) + torch.unsqueeze(torch.unsqueeze(self.layer.bias, 0), 0)
       stabilizer = 1e-8 * np.where(Zs >= 0, torch.ones_like(Zs), torch.ones_like(Zs) * -1)
       Zs += torch.Tensor(stabilizer)

       return torch.sum((Z / Zs) * torch.unsqueeze(self.R, 1), dim=2)

   def _alphabeta_lrp(self, R, alpha):
       self.R = R
       beta = 1 - alpha

       Z = torch.unsqueeze(torch.transpose(self.layer.weight, 0, 1), 0) * torch.unsqueeze(self.input, -1)
       if not alpha == 0:
           Zp = np.where(Z > 0, Z, torch.zeros_like(Z))
           Zp = where(torch.ByteTensor(Z > 0), Z, torch.zeros_like(Z))
           term2 = torch.unsqueeze()


       Zs = tf.expand_dims(tf.reduce_sum(Z, 1), 1) + tf.expand_dims(tf.expand_dims(self.biases, 0), 0)
       stabilizer = 1e-8 * (tf.where(tf.greater_equal(Zs, 0), tf.ones_like(Zs, dtype=tf.float32),
                                     tf.ones_like(Zs, dtype=tf.float32) * -1))
       Zs += stabilizer