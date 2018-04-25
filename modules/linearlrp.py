import torch
import torch.nn as nn
import torch.nn.functional as F

class linearlrp(nn.Linear):

    # def __init__(self):
    #  self.lrpopts=lrpoptions()

    def __init__(self, source=None):
        if source is not None:
            self.__dict__.update(source.__dict__)

            if type(source) != type(self):
                self.lrpopts = lrpoptions()

    class computations(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            # reimplement forward, stash by ctx.save what you need

            ctx.save_for_backward(input, weight, bias)
            output = input.mm(weight.t())
            if bias is not None:
                output += bias.unsqueeze(0).expand_as(output)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            print('hello')
            input, weight, bias = ctx.saved_variables
            grad_input = grad_weight = grad_bias = None

            '''
            if ctx.needs_input_grad[0]:
              grad_input = grad_output.mm(weight)
            if ctx.needs_input_grad[1]:
              grad_weight = grad_output.t().mm(input)
            if bias is not None and ctx.needs_input_grad[2]:
              grad_bias = grad_output.sum(0).squeeze(0)
            '''

            # not sure what to return as format
            return grad_input, grad_weight, grad_bias

    def forward(self, x):
        return self.computations.apply(x, self.weight, self.bias)
