import torch
import torch.nn as nn
import torch.nn.functional as F
#
# from my_models.PatchTST.patch import PatchTST
# from my_models.PatchTST.posotion_encod import my_positional_encoding
# from my_models.Vision.plot_deformable import plot_layer
from my_models.tools.Deformable_attn import deformable_LKA_Attention
from my_models.tools.Patch_embed import TokenEmbedding, PositionalEmbedding, PatchEmbedding
from my_models.tools.def_depth_conv import Def_dep
from my_models.tools.multi_deformable_conv import Multi_Def_Conv
import matplotlib.pyplot as plt

'''
pre_train 替换为 pre_train
'''
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

        #start
        # mask_data = x_patch_masked[0][0].cpu()
        # # print(mask_data)
        # def_layer = repres[0][0]
        # def_layer = def_layer.cpu().detach().numpy()
        # if i == 0 or i == 16 or i == 32:
        #     name = "Deformable_exc" + str(epoch) + "_" + str(i)
        #     plt.subplot(1, 2, 1)
        #     plt.title('Original Image')
        #     plt.imshow(mask_data)
        #     plt.axis('off')
        #
        #     plt.subplot(1, 2, 2)
        #     plt.title('Dilated Convolution Output')
        #     plt.imshow(def_layer)  # (256,256,64)
        #     plt.axis('off')
        #     plt.savefig(
        #         r'C:\Users\11848\Desktop\PatchForecasting\my_models\Vision\res\\' + name + '.pdf',
        #         dpi=300)
        # end
        #
        all_fea = repres.permute(0, 1, 3, 2).reshape(b, -1, n)
        output = self.proj(all_fea.permute(0,2,1)).permute(0, 2, 1)

        return output

    def forecast(self, x_enc, epoch, i):
        return self.encoder(x_enc, epoch, i)

    ''''
    def patch_Sequence(self, x_in, ori_data, mask_ratio):
        _, H, W = x_in.shape
        x_enc = x_in.unsqueeze(1)  # (32,1,96,8)
        seq_unf = self.unfold(x_enc)
        s2p = seq_unf.permute(0, 2, 1)
        # start
        ori_data = ori_data.unsqueeze(1)
        ori_unf = self.unfold(ori_data)
        ori_s2q = ori_unf.permute(0,2,1)
        # end
        bs, L, D = s2p.shape
        # mask_ratio = 0.5
        len_keep = int(L * (1 - mask_ratio))  # 每次mask是以patch为单位，保留的数量
        noise = torch.rand(bs, L)  # noise in [0, 1], bs x L
        # sort noise for each sample, argsort只返回索引
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]  # ids_keep: [bs x len_keep x nvars]
        x_kept = torch.gather(s2p, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))  # 按patch获取。gathet: https://zhuanlan.zhihu.com/p/352877584  # x_kept: [bs x len_keep x nvars  x patch_len]
        # start
        ori_kept = torch.gather(ori_s2q, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # end
        # removed x
        x_removed = torch.zeros(bs, L - len_keep, D)  # [64,17,7,12]# x_removed: [bs x (L-len_keep) x nvars x patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len] [64,42,7,12]
        # start
        ori_k = torch.cat([ori_kept, x_removed], dim=1)
        # end
        # combine the kept part and the removed one,
        unf_masked = torch.gather(x_, dim=1,
                                index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x nvars x patch_len]

        ori_masked = torch.gather(ori_k, dim=1,
                                index=ids_restore.unsqueeze(-1).repeat(1, 1, D))  # x_masked: [bs x num_patch x nvars x patch_len]



        fold = nn.Fold(output_size=(H,W), kernel_size=self.kernel, stride=self.kernel)
        f_masked = unf_masked.permute(0, 2, 1)
        x_enc_masked = fold(f_masked).squeeze(1)

        # start
        ori_masked = ori_masked.permute(0, 2, 1)
        ori_enc_masked = fold(ori_masked).squeeze(1)
        # end

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L])  # (64,42,7)  # mask: [bs x num_patch x nvars]
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask_ush = torch.gather(mask, dim=1, index=ids_restore)  # mask对应x_masked.
        mask_ = mask_ush.unsqueeze(-1).expand_as(unf_masked).permute(0, 2, 1)
        mask_index = fold(mask_).squeeze(1)

        return x_enc_masked, ori_enc_masked, mask_index  # The input data has been masked; The masked data index

    def create_group(self, x_enc_masked, patch_len, stride):
        """
        xb: [bs x pre_train x n_vars]
        """
        pre_train = x_enc_masked.shape[1]
        num_patch = self.num_group
        tgt_len = patch_len + stride * (num_patch - 1)
        s_begin = pre_train - tgt_len

        xb = x_enc_masked[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]真正参与计算的矩阵，这也是目标值y
        x_patch_masked = xb.unfold(dimension=1, size=patch_len, step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
        return x_patch_masked, num_patch, xb
    '''

    def forward(self, x_enc, epoch, i):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, epoch,i)
            return dec_out
        return None
