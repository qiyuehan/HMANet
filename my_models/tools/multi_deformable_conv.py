import torch
import torch.nn as nn

#HDC
from my_models.tools.deformable import Def_Conv
# modify
class Multi_Def_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates):
        super(Multi_Def_Conv, self).__init__()

        # 创建多个卷积层，每个层有不同的空洞率
        self.conv_layers = nn.ModuleList()
        # self.conv2D = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2D = nn.Conv2d(in_channels*3, in_channels, 3,1,1)
        for dilation in dilation_rates:
            # self.conv_layers.append(
            #     nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=dilation, dilation=dilation)
            # )

            self.conv_layers.append(
                    Def_Conv(in_channels, in_channels, kernel_size=3, dilation_rate=dilation)  # Deformable Conv
                )
    #
    def forward(self, x):
        # 对输入进行多尺度可变卷积并将结果相加
        output = []
        for conv_layer in self.conv_layers:
            output.append(conv_layer(x))
        cat_out = torch.cat(output,dim=1)  # 融合不同尺度的信息+不同维度之间的相关性信息
        output = self.conv2D(cat_out)

        # 直接相加 mse imporve 0.005
        # output = 0
        # for conv_layer in self.conv_layers:
        #     output += conv_layer(x)


        return output

if __name__ == '__main__':

    in_channels = 3
    out_channels = 64
    dilation_rates = [1, 2, 5]

    # 创建多尺度空洞卷积层

    msac = Multi_Def_Conv(in_channels, out_channels, dilation_rates)

    # 输入图像大小为(N, C, H, W)
    input_image = torch.randn(1, in_channels, 64, 64)

    # 前向传播
    output = msac(input_image)





