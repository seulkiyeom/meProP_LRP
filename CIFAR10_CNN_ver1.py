# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable, Function
from modules.data import get_mnist, get_cifar10
# from modules.module import Module
# from modules.linearlrp import linearlrp
from modules.Linear import Linear
from modules.Convolution import Conv2d
from modules.Maxpool import MaxPool2d
from modules.Avgpool import AvgPool2d
from modules.Softmax import LogSoftmax
from modules.Relu import ReLU
from modules.util import plot_relevances, plot_relevances_3d
from collections import OrderedDict

layer_name = []

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1001, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
    parser.add_argument('--save-dir', type=str, default='cifar_model', metavar='N',
                        help='saved directory')
    parser.add_argument('--save-model', type=bool, default=False, metavar='N',
                        help='Save the trained model')
    parser.add_argument('--reload-model', type=bool, default=False, metavar='N',
                        help='Restore the trained model')
    parser.add_argument('--relevance', type=bool, default=False, metavar='N',
                        help='Compute relevances')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    # MNIST Dataset
    # Data Acquisition
    train_dataset, test_dataset = get_cifar10()

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
        self.input = input[0]
        self.output = output.data


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # 3 input image channel, 6 output channels, 5x5 square convolution
            # kernel
            self.layer = nn.Sequential(OrderedDict([
                ('conv1', Conv2d(3, 6, 5)),
                ('relu1', ReLU()),
                ('mp1', MaxPool2d(2)),
                ('conv2', Conv2d(6, 16, 5)),
                ('relu2', ReLU()),
                ('mp2', MaxPool2d(2))
            ]))
            self.fc_layer = nn.Sequential(OrderedDict([
                ('fc1', Linear(16 * 5 * 5, 120)),
                ('fc_relu1', ReLU()),
                ('fc2', Linear(120, 84)),
                ('fc_relu2', ReLU()),
                ('fc3', Linear(84, 10)),
                ('sm', LogSoftmax(dim=1))

            ]))

        def forward(self, x):
            in_size = x.size(0)
            x = self.layer(x)
            x = x.view(in_size, -1)  # flatten the tensor
            x = self.fc_layer(x)
            return x


        def forward_hook(self):
            # For Forward Hook
            global layer_name
            for name, module in self.layer.named_children():
                self.hook = module.register_forward_hook(printnorm)
                layer_name.append(name)
            for name, module in self.fc_layer.named_children():
                self.hook = module.register_forward_hook(printnorm)
                layer_name.append(name)


        def lrp(self, R):
            for cur_layer in layer_name[::-1]:
                find = False
                for name, module in self.fc_layer.named_children():  # 접근 방법
                    if name is cur_layer:
                        # print(name)
                        R = module.lrp(R, args.relevance_method, 1e-8)
                        # self.hook.remove()
                        find = True
                    if find:
                        break

                for name, module in self.layer.named_children():  # 접근 방법
                    if name is cur_layer:
                        # print(name)
                        R = module.lrp(R, args.relevance_method, 1e-8)
                        # self.hook.remove()
                        find = True
                    if find:
                        break

            return R

    model = Net()

    if args.relevance:
        model.forward_hook()

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

            # model.lrp(R)
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
        R_tot = []
        data_tot = []
        if args.reload_model:
            model.load_state_dict(torch.load('./' + args.save_dir + '.pth', map_location=lambda storage, loc: storage))

        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, requires_grad=True), Variable(target)
            output = model(data)

            # Explanation
            if args.relevance:
                R = output
                R_out = model.lrp(R)
                # R_tot = torch.cat((R_tot, R_out))
                R_tot = torch.cat((Variable(torch.FloatTensor(R_tot)), R_out.data))
                data_tot = torch.cat((Variable(torch.FloatTensor(data_tot)), data.data))

            test_loss += F.nll_loss(output, target, size_average=False).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

        if args.relevance:
            plot_relevances_3d(R_tot, data_tot, image_show=True, image_save=True)

    for epoch in range(1, args.epochs + 1):
        train(epoch)

        test() #매 epoch 마다 테스트 하고싶으면

    test() #훈련되어 있는 training model에 test만


if __name__ == "__main__":
    main()