import torch.nn as nn

from my_models.tools.deformable import Def_Conv


class Def_dep(nn.Module):
    def __init__(self, in_size, out_size, ker_size=3, stride=1, padding=1, dilation=0):
        super(Def_dep, self).__init__()
        self.depth_conv = Def_Conv(in_c=in_size,
                                     out_c=in_size,
                                     kernel_size=3,
                                     dilation_rate=1,
                                     groups=in_size)
        self.point_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=out_size,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
        # self.conv = nn.Conv2d(in_size, out_size, kernel_size=ker_size, stride=stride, padding=padding, groups=in_size)
        self.bn = nn.BatchNorm2d(out_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x1 = self.depth_conv(x)
        x2 = self.point_conv(x1)
        x3 = self.relu(x2)
        x4 = self.bn(x3)
        return x4

if __name__ == '__main__':
    import torch
    x = torch.randn(32,5,96,7)
    m = Def_dep(5, 10)
    o = m(x)
