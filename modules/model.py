from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# from modules import Linear


class Net(nn.Module):
    def __init__(self, k, dropout=None):
        super(Net, self).__init__()
        self.k = k
        self.dropout = dropout
        self.model = nn.Sequential(self._create(k, dropout))

    def _create(self, k, dropout=None):
        d = OrderedDict()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        # self.fc = nn.Linear(320, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1)  # flatten the tensor
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


        for i in range(layer):
            if i == 0:
                d['linear' + str(i)] = Linear(784, hidden, k, self.unified)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
            elif i == layer - 1:
                d['linear' + str(i)] = Linear(hidden, 10, 0, self.unified)
            else:
                d['linear' + str(i)] = Linear(hidden, hidden, k, self.unified)
                d['relu' + str(i)] = nn.ReLU()
                if dropout:
                    d['dropout' + str(i)] = nn.Dropout(p=dropout)
        return d