import torch
from torch import nn
from torch.nn import functional as F


#deepconv
# from pytorch_project.tool.tools import modify_deep


class Mul_avg_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, d_model):
        modules = [
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.LayerNorm(d_model),
            nn.ReLU()
        ]
        super(Mul_avg_conv, self).__init__(*modules)

class Avg_Pooling(nn.Sequential):
    def __init__(self, in_channels, out_channels, d_model):
        super(Avg_Pooling, self).__init__()
        self.ada_avag = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.lay_norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        size = x.shape[-1:]
        x = self.ada_avag(x)
        x = self.conv(x)
        x = F.interpolate(x, size=size, mode='linear', align_corners=False)
        x = self.lay_norm(x)
        x = self.relu(x)
        return x

class modify_deep(nn.Module):
    def __init__(self, in_size, out_size, ker_size=3, stride=1, padding=1, dilation=0):
        super(modify_deep, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels=in_size,
                                    out_channels=in_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
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

    # 深度可分离卷积
    def forward(self, x):
        x1 = self.depth_conv(x)
        x2 = self.point_conv(x1)
        x3 = self.relu(x2)
        x4 = self.bn(x3)
        return x4


# depthwise + spartial
class mul_conv(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels, out_dim=7):
        super(mul_conv, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.LayerNorm(out_dim),
            nn.ReLU()))

        # 多尺度空洞卷积moe
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(Mul_avg_conv(in_channels, out_channels, rate, out_dim))
        # 池化
        modules.append(Avg_Pooling(in_channels, out_channels, out_dim))

        self.convs = nn.ModuleList(modules)
        # 时间
        self.temporal = nn.Linear(len(self.convs)*in_channels, in_channels)
        # 空间
        self.spatial = nn.Conv2d(len(self.convs), 1, kernel_size=3, padding=1)
        self.mod = modify_deep(len(self.convs), len(self.convs)).to(device=self.device)

    def forward(self, x_per):
        res_per = []
        x_per = torch.tensor(x_per, dtype=torch.float)
        for conv in self.convs:
            res_per.append(conv(x_per))
        # 多空洞卷积
        temporal_fea = torch.cat(res_per, dim=1)
        t_fea = self.temporal(temporal_fea.permute(0,2,1))

        stack_per = torch.stack(res_per, dim=1)
        s_fea = self.spatial(stack_per).squeeze()
        res = t_fea.permute(0,2,1) + s_fea
        # 去除两个模块
        # res = x_per
        return res


class mul_conv_layer(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels, out_dim=7):
        super(mul_conv_layer, self).__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.Linear = nn.Linear(in_channels, int(in_channels/2)).to(self.device)

        modules = []
        # 1*1 卷积
        modules.append(nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            # nn.Linear(in_channels, out_channels, bias=False),
            nn.LayerNorm(out_dim),
            nn.ReLU()))

        # 多尺度空洞卷积moe
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(Mul_avg_conv(in_channels, out_channels, rate, out_dim))
        # 池化
        modules.append(Avg_Pooling(in_channels, out_channels, out_dim))

        self.convs = nn.ModuleList(modules)
        # 时间
        self.temporal = nn.Linear(in_channels, in_channels)
        # 空间
        self.spatial = nn.Conv2d(len(self.convs), 1, kernel_size=3, padding=1)


    def forward(self, low, high):
        n, _, _ = low.shape
        res_per = []
        low_T = low.permute(0,2,1)
        high_T = high.permute(0,2,1)
        x_per = torch.concat([low_T, high_T], dim=1)
        x_per = torch.tensor(x_per, dtype=torch.float)
        for conv in self.convs:
            res_per.append(conv(x_per))
        # 多空洞卷积
        # temporal_fea = torch.cat(res_per, dim=1)
        temporal_fea=0
        for i in range(len(res_per)):
            temporal_fea += res_per[i]
        # temporal_fea = res_per[0] + res_per[1] + res_per[2] + res_per[3] + res_per[4] + res_per[5]
        t_fea = self.temporal(temporal_fea.permute(0, 2, 1))

        stack_per = torch.stack(res_per, dim=1)
        s_fea = self.spatial(stack_per).squeeze()
        # sum_res = t_fea.permute(0, 2, 1) + s_fea

        return t_fea.permute(0, 2, 1).reshape(n,-1), s_fea.reshape(n,-1)
        # res = self.Linear(sum_res.permute(0, 2, 1)).permute(0, 2, 1)
        # return low, res.permute(0, 2, 1)

if __name__ == '__main__':
    x = torch.randn(32, 96, 32).cuda()
    pool = mul_conv(96, [ 4, 8, 16, 32], 96, 32).cuda()
    a = pool(x)
    print(a.shape)
