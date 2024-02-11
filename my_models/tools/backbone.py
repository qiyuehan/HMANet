import torch
import torch.nn as nn
from my_models.tools.multi_deformable_conv import Multi_Def_Conv

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pre_train = configs.pre_train
        self.enc_in = configs.enc_in

        self.mask_ratio = configs.mask_rate
        self.kernel = configs.kernel_size
        self.patch_len = configs.patch_len
        self.num_group = (max(configs.pre_train, self.patch_len) - self.patch_len) // self.patch_len + 1
        self.mul_deformable_conv = Multi_Def_Conv(in_channels=self.num_group, out_channels=self.num_group,
                                                  dilation_rates=configs.dilation_rate)

        self.ff = nn.Sequential(nn.Conv2d(self.num_group, self.num_group, kernel_size=3, padding=1, stride=1),
                                nn.Dropout(configs.dropout) )

        self.bn = nn.BatchNorm2d(self.num_group)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(configs.dropout)
        self.proj = nn.Linear(self.pre_train, self.pre_train)
        self.share_p = nn.Conv2d(self.num_group, self.num_group, 1)
        self.block_linear = nn.Linear(self.num_group, self.num_group)


    def encoder(self, x_patch_masked,epoch, i):

        b, num_p, block_len, n = x_patch_masked.shape
        x_block_mask = x_patch_masked.permute(0,2,3,1)
        group_inner = self.block_linear(x_block_mask).permute(0,3,1,2)
        mul_deformabel = self.mul_deformable_conv(x_patch_masked)
        repres = group_inner + group_inner * torch.softmax(mul_deformabel, dim=1)
        repres = self.ff(repres)
        all_fea = repres.reshape(b, -1, n)
        output = self.proj(all_fea.permute(0,2,1)).permute(0, 2, 1)
        return output

    def forecast(self, x_enc, epoch, i):
        return self.encoder(x_enc, epoch, i)

    def forward(self, x_enc, epoch, i):
        dec_out = self.forecast(x_enc, epoch,i)
        return dec_out

