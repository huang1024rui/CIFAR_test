# -*- coding: utf-8 -*—
# Date: 2022/2/25
# Time: 8:28
# Author: 
import torch
from torch import nn


class HQR_CIFAR10(nn.Module):
    def __init__(self, dim, layer):
        super(HQR_CIFAR10, self).__init__()
        self.HQR = nn.ModuleList()
        self.HQR.append(self.conv_block(dim, layer[0], layer[1]))   # 1
        self.HQR.append(self.conv_block(dim, layer[1], layer[2]))   # 2
        self.HQR.append(self.conv_block(dim, layer[2], layer[3]))   # 3
        self.HQR.append(nn.Flatten())                               # 4
        self.HQR.append(nn.Linear(layer[4], layer[5]))              # 5
        self.HQR.append(nn.Linear(layer[5], layer[6]))              # 6

    def conv_block(self, dim, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
         conv_fn = getattr(nn, "Conv{0}d".format(dim))
         bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
         maxp_fn = getattr(nn, "MaxPool{0}d".format(dim))
         layer = nn.Sequential(
             conv_fn(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
             maxp_fn(2),
             bn_fn(out_channels),
             nn.LeakyReLU((0.2))
         )
         return layer

    def forward(self, input):
        x = self.HQR[0](input)
        x1 = self.HQR[1](x)
        x2 = self.HQR[2](x1)
        x3 = self.HQR[3](x2)
        x4 = self.HQR[4](x3)
        output = self.HQR[5](x4)
        return output

if __name__ == '__main__':
    layer = [3, 32, 32, 64, 64*4*4, 64, 10]
    HQR = HQR_CIFAR10(2, layer)
    print(HQR)
    input = torch.ones((64, 3, 32, 32))
    output = HQR(input)
    print(output.shape)
