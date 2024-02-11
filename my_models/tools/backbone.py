import torch
import torch.nn as nn
import torch.nn.functional as F

from my_models.tools.Deformable_attn import deformable_LKA_Attention
from my_models.tools.Patch_embed import TokenEmbedding, PositionalEmbedding, PatchEmbedding
from my_models.tools.def_depth_conv import Def_dep
from my_models.tools.multi_deformable_conv import Multi_Def_Conv
import matplotlib.pyplot as plt


class SeqToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    def forward(self, x_enc):
        assert len(x_enc.size()) == 4
        s2p = self.unfold(x_enc)
        s2p = s2p.permute(0, 2, 1)
        return s2p

class Model(nn.Module):

    def __init__(self, configs, chunk_size=24):
        """
        chunk_size: int, reshape T into [num_chunks, chunk_size]
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pre_train = configs.pre_train

        # self.d_model = configs.d_model
        self.enc_in = configs.enc_in

        self.mask_ratio = configs.mask_rate
        self.kernel = configs.kernel_size
        self.patch_len = configs.patch_len
        self.num_group = (max(configs.pre_train, self.patch_len) - self.patch_len) // self.patch_len + 1
        self.mul_deformable_conv = Multi_Def_Conv(in_channels=self.num_group, out_channels=self.num_group,
                                                  dilation_rates=configs.dilation_rate)

        self.ff = nn.Sequential(nn.Conv2d(self.num_group, self.num_group, kernel_size=3, padding=1, stride=1),
                                # nn.LeakyReLU()
                                nn.Dropout(configs.dropout),
                                # nn.BatchNorm2d(self.num_group)

                                )
        # self.conv2d = nn.Conv2d(self.num_group*2, self.num_group, kernel_size=3, padding=1, stride=1)

        self.bn = nn.BatchNorm2d(self.num_group)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(configs.dropout)

        self.proj = nn.Linear(self.pre_train, self.pre_train)
        # self.proj = nn.Linear(configs.d_model, self.kernel[0])


        self.patch_embedding = PatchEmbedding(
                     configs.d_model, patch_len=self.kernel[0],stride=1, padding=0, dropout=configs.dropout)

        # self.W_pos = my_positional_encoding(learn_pe=True, q_len=self.num_group, d_model=self.kernel[0])  # (42, 128)

        self.share_p = nn.Conv2d(self.num_group, self.num_group, 1)
        self.block_linear = nn.Linear(self.num_group, self.num_group)


    def encoder(self, x_patch_masked,epoch, i):

        b, num_p, block_len, n = x_patch_masked.shape
        x_block_mask = x_patch_masked.permute(0,2,3,1)
        group_inner = self.block_linear(x_block_mask).permute(0,3,1,2)
        # group_inner = self.share_p(x_block_mask)
        # Deformable conv
        mul_deformabel = self.mul_deformable_conv(x_patch_masked)  # [32, 88, 8, 12]
        # repres = group_inner + x_patch_masked * torch.softmax(mul_deformabel, dim=1) #[32, 88, 8, 12]
        repres = group_inner + group_inner * torch.softmax(mul_deformabel, dim=1) #[32, 88, 8, 12]

        repres = self.ff(repres)

        all_fea = repres.reshape(b, -1, n)
        output = self.proj(all_fea.permute(0,2,1)).permute(0, 2, 1)
        return output

    def encoder_ori(self, x_patch_masked,epoch, i):

        b, num_p,n, patch_len = x_patch_masked.shape
        group_inner = self.share_p(x_patch_masked)

        '''
        # Use Linear
        x_in = x_patch_masked.reshape(b, num_p, -1).permute(0, 2, 1)
        group_inner = self.share_p(x_in)  # Conv2d(60, 120, kernel_size=(1, 1), stride=(1, 1))
        group_inner = group_inner.permute(0, 2, 1).reshape(b, self.num_group, n, patch_len)
        '''
        # Deformable conv
        mul_deformabel = self.mul_deformable_conv(x_patch_masked)  # [32, 88, 8, 12]

        # mul_ori_deformable = self.mul_deformable_conv(ori_patch_masked)
        # Vision deformable start
        # print(x_patch_masked[0][0])
        # plot_layer(mask_data[0], 'Mask Data')
        # def_layer = mul_deformabel[0][0] + mul_deformabel[0][1]
        # def_layer = def_layer.cpu().detach().numpy()
        # if i == 0 or i == 7 or i == 18:
        #     name = "Deformable_ETTh2" + str(epoch) + "_" + str(i)
        #     plot_layer(def_layer, name)
        # # end

        # representation
        # repres = group_inner + group_inner * torch.softmax(mul_deformabel, dim=1) #[32, 88, 8, 12]
        repres = group_inner + x_patch_masked * torch.softmax(mul_deformabel, dim=1) #[32, 88, 8, 12]

        repres = self.ff(repres)

        all_fea = repres.permute(0, 1, 3, 2).reshape(b, -1, n)
        output = self.proj(all_fea.permute(0,2,1)).permute(0, 2, 1)

        return output

    def forecast(self, x_enc, epoch, i):
        return self.encoder(x_enc, epoch, i)


    def forward(self, x_enc, epoch, i):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, epoch,i)
            return dec_out
        return None
