import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.nn.modules.utils import _pair
from torch.autograd import Function


class SEConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(SEConv2d, self).__init__(in_channels,
                                       out_channels,
                                       kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       dilation=dilation,
                                       groups=groups,
                                       bias=bias)

    def forward(self, input):
        weight = self.weight.detach()
        output = F.conv2d(input, weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)

        return output