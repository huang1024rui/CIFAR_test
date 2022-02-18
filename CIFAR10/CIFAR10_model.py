# -*- coding: utf-8 -*—
# Date: 2022/2/16 0016
# Time: 8:56
# Author: 
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, BatchNorm2d, ReLU, AvgPool2d


class HQR_CIFAR10(nn.Module):
    def __init__(self):
        super(HQR_CIFAR10, self).__init__()
        self.HQR_CIFAR10 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(64*4*4, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.HQR_CIFAR10(input)
        return output



if __name__ == '__main__':
    HQR_CIFAR10 = HQR_CIFAR10()
    input = torch.ones((64, 3, 32, 32))
    output = HQR_CIFAR10(input)
    print(output.shape)
