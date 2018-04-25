# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, Function
from modules.data import get_mnist
from modules.module import GuidedBackpropRelu, GuidedReluModel
from modules.linearlrp import linearlrp

from modules.save_model import save_model

from collections import OrderedDict

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--batch-size', type=int, default=99, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--relevance-method', type=str, default='simple', metavar='N',
                        help='relevance methods: simple/eps/w^2/alphabeta')
    parser.add_argument('--save-dir', type=str, default='model', metavar='N',
                        help='saved directory')
    parser.add_argument('--save-model', type=bool, default=True, metavar='N',
                        help='Save the trained model')
    parser.add_argument('--reload-model', type=bool, default=False, metavar='N',
                        help='Restore the trained model')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # MNIST Dataset
    # Data Acquisition
    train_dataset, test_dataset = get_mnist()

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.test_batch_size,
                                              shuffle=False, **kwargs)

    def printnorm(self, input, output):
        # input is a tuple of packed inputs
        # output is a Variable. output.data is the Tensor we are interested

        # print('Inside ' + self.__class__.__name__ + ' forward')
        # print('')
        # print('input: ', type(input))
        # print('input[0]: ', type(input[0]))
        # print('output: ', type(output))
        # print('')
        # print('input size:', input[0].size())
        # print('output size:', output.data.size())
        # print('output norm:', output.data.norm())
        self.input = input[0]
        self.output = output.data

    class MyFunction(Function):

        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            output = torch.sign(input)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            # saved tensors - tuple of tensors, so we need get first
            input, = ctx.saved_variables
            grad_output[input.ge(1)] = 0
            grad_output[input.le(-1)] = 0
            return grad_output

    my_func = MyFunction.apply

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.layer = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(1, 6, 5)),
                ('relu1', nn.ReLU()),
                ('mp1', nn.MaxPool2d((2, 2))),
                ('conv2', nn.Conv2d(6, 16, 5)),
                ('relu2', nn.ReLU()),
                ('mp2', nn.MaxPool2d((2, 2)))
            ]))
            self.fc_layer = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(256, 120)),
                ('fc_relu1', nn.ReLU()),
                ('fc2', nn.Linear(120, 84)),
                ('fc_relu2', nn.ReLU()),
                ('fc3', nn.Linear(84, 10)),
            ]))

        def forward(self, x):
            in_size = x.size(0)
            x = self.layer(x)
            x = x.view(in_size, -1)  # flatten the tensor
            x = self.fc_layer(x)
            return F.log_softmax(x, dim=1)

    model = Net()

    # For Forward Hook
    for name, module in model.layer.named_children():
        print(name)
        module.register_forward_hook(printnorm)
    for name, module in model.fc_layer.named_children():
        print(name)
        module.register_forward_hook(printnorm)

    if args.cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()

            # Forward Pass
            output = model(data)
            R = output
            loss = F.nll_loss(output, target)

            for name, module in model.layer.named_children(): #접근 방법
                module.output.shape

            # Backward Pass
            loss.backward()  # Calculation of Gradient
            # param_model = list(model.parameters()) #to show all W in the model


            # for W in reversed(list(model.parameters())):
            #     W.grad.data
            #     W.data

            # optimizer.step() #Weight Parameter Update using Gradient
            for W in model.parameters():
                W.data = W.data - args.lr * W.grad.data

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))

        if args.save_model:
            torch.save(model.state_dict(), './' + args.save_dir + '.pth')

    def test():
        if args.reload_model:
            model.load_state_dict(torch.load('./' + args.save_dir + '.pth', map_location=lambda storage, loc: storage))

        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)

            # Explanation
            R = output

            test_loss += F.nll_loss(output, target, size_average=False).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    for epoch in range(1, args.epochs + 1):
        train(epoch)

        test()


if __name__ == "__main__":
    main()