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
from torch.autograd import Variable
from modules.data import get_mnist
from modules.module import LRP

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
    parser.set_defaults(unified=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # MNIST Dataset
    #Data Acquisition
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

        self.output = output.data


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 1 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.conv1 = nn.Conv2d(1, 6, 5)
            self.conv2 = nn.Conv2d(6, 16, 5)
            # an affine operation: y = Wx + b
            self.fc1 = nn.Linear(256, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)
            self.drop = nn.Dropout2d()

        def forward(self, x):
            m = list()
            in_size = x.size(0)
            m.append(F.relu(self.conv1(x))) #m[0]: Conv + ReLu
            m.append(F.max_pool2d(m[0], 2)) #m[1]: MaxPool
            # x = F.dropout(x, p=0.8)
            m.append(F.relu(self.conv2(m[1]))) #m[2]: Conv + ReLu
            m.append(F.max_pool2d(m[2], 2)) #m[3]: MaxPool
            m[3] = m[3].view(in_size, -1)  # flatten the tensor

            # x = F.dropout(x, p=0.8)
            m.append(F.relu(self.fc1(m[3]))) #m[4]: Fully-Connected
            m.append(F.relu(self.fc2(m[4]))) #m[5]: Fully-Connected
            # x = F.dropout(x, training=self.training)
            m.append(self.fc3(m[5])) #m[6]: Fully-Connected

            return F.log_softmax(m[6], dim=1), m

    model = Net()
    if args.cuda:
        model.cuda()

    # for name in model._modules:
    #     activations = SaveFeatures(model._modules.get(name))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    def train(epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()

            model.conv2.register_forward_hook(printnorm)
            model.conv1.register_forward_hook(printnorm)

            # print(model.conv1.output)
            #
            # model.conv1.output.shape
            # model_out[0].data.shape

            #model.conv1.output == model_out[0].data 이게 같지 않음!!


            #Forward Pass
            output, model_out = model(data)
            R = output
            loss = F.nll_loss(output, target)
            #################
            #Backward Pass
            loss.backward() #Calculation of Gradient
            # param_model = list(model.parameters()) #to show all W in the model

            #
            # for idx, m in enumerate(model.named_modules()):
            #     print(idx, '->', )
            #
            # for W in reversed(list(model.parameters())):
            #     R = LRP(W, R, args.relevance_method, 1e-8)
            #     # print m.name + '::',
            #     print(W.grad.data.shape)
            #
            #     W.grad.data
            #     W.data

            # optimizer.step() #Weight Parameter Update using Gradient
            for W in model.parameters():
                # print(W.grad.data)
                W.data = W.data - args.lr * W.grad.data
            #################

            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output, model_out = model(data)
            R = output
            # sum up batch loss
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