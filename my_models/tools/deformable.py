import torch.nn as nn
import torch
import numpy as np
import torchvision

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        self.conv_offset = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([18, 3, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset)

        self.conv_mask = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([9, 3, 3, 3]) + np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask)  # 初始化为0.5

    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x))
        out = torchvision.ops.deform_conv2d(input=x, offset=offset,
                                            weight=self.conv.weight,
                                            mask=mask, padding=(1, 1))
        return out


import torchvision.ops as ops
class Def_Conv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, dilation_rate=1, groups=1):
        super(Def_Conv, self).__init__()
        self.split_size = (2 * kernel_size * kernel_size, kernel_size * kernel_size)
        self.conv_offset = nn.Conv2d(in_c, 3 * kernel_size * kernel_size, kernel_size, padding=dilation_rate, dilation=dilation_rate)
        self.conv_deform = ops.DeformConv2d(in_c, out_c, kernel_size, padding=dilation_rate, dilation=dilation_rate, groups=in_c)

        # initialize
        nn.init.constant_(self.conv_offset.weight, 0)
        nn.init.constant_(self.conv_offset.bias, 0)
        nn.init.kaiming_normal_(self.conv_deform.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        offset, mask = torch.split(self.conv_offset(x), self.split_size, dim=1)
        mask = torch.sigmoid(mask)
        def_out = self.conv_deform(x, offset, mask)
        return def_out

