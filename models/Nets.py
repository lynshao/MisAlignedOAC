#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import pdb


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x


class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNCifar1(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # # nn.Conv2d(in_channels, out_channels, kernel_size, ...)
        self.kernel_size = 5;
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=self.kernel_size ,stride=1,padding=int((self.kernel_size-1)/2))
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size ,stride=1,padding=int((self.kernel_size-1)/2))
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=self.kernel_size ,stride=1,padding=int((self.kernel_size-1)/2))
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, args.num_classes)

        ########################################################### More Powerful model
        # self.base = nn.Sequential(
        #     nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(True)
        # )
        # self.fc1 = nn.Linear(128*8*8, 2048)
        # self.fc2 = nn.Linear(2048, 128)
        # self.fc3 = nn.Linear(128, args.num_classes)

    def forward(self, x):
        # # # input shape = [batchsize, channel=3. length=32, width=32]
        x = self.pool1(F.relu(self.conv1(x))) # shape = [batchsize, channel=32. length=16, width=16]
        x = self.pool2(F.relu(self.conv2(x))) # shape = [batchsize, channel=64. length=8, width=8]
        x = x.view(-1, 64 * 8 * 8) # shape of output [10, 400]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        ########################################################### More Powerful model
        # pdb.set_trace()
        # x = self.base(x)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = x.view(-1, 128*8*8)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)


        return x

class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x