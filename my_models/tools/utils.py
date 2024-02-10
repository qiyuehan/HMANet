from thop import profile

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import get_stamp
from my_models.tools.patch_basic import Patch_Basic

from utils.gpu_mem_track import MemTracker
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
import time

warnings.filterwarnings('ignore')

# gpu_tracker = MemTracker()

# Representation , PreTrain, Forecasting
class Patch_Rep(Patch_Basic):
    def __init__(self, args):
        super(Patch_Rep, self).__init__(args)
        self.patch_size = args.patch_len
        self.kernel = args.kernel_size
        self.unfold = nn.Unfold(kernel_size=self.kernel , stride=self.kernel)
        self.fold = nn.Fold(output_size=(args.pre_train, args.enc_in), kernel_size=self.kernel, stride=self.kernel)
        self.num_group = (max(args.pre_train, self.patch_size) - self.patch_size) // self.patch_size + 1
        self.model_path = args.checkpoints
        self.model_name = args.model_id
        self.advice = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    def patch_loss(self, preds, target, mask):
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len]
        """

        loss = (preds.cpu() - target.cpu()) ** 2
        mask = mask.cpu()
        #  ablation，当先patch在mask的时候打开
        # loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss


    def ablation_loss(self, preds, target, mask):
        """
        preds:   [bs x num_patch x n_vars x patch_len]
        targets: [bs x num_patch x n_vars x patch_len]
        """

        loss = (preds.cpu() - target.cpu()) ** 2
        mask = mask.cpu()
        #  ablation，当先patch在mask的时候打开
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def patch_Sequence(self, x_in, ori_data, mask_ratio):
        _, H, W = x_in.shape
        x_enc = x_in.unsqueeze(1)  # (32,1,96,8)
        seq_unf = self.unfold(x_enc)
        s2p = seq_unf.permute(0, 2, 1)
        # start
        ori_data = ori_data.unsqueeze(1)
        ori_unf = self.unfold(ori_data)
        ori_s2q = ori_unf.permute(0, 2, 1)
        # end
        bs, L, D = s2p.shape
        # mask_ratio = 0.5
        len_keep = int(L * (1 - mask_ratio))  # 每次mask是以patch为单位，保留的数量
        noise = torch.rand(bs, L).to(self.device)   # noise in [0, 1], bs x L
        # sort noise for each sample, argsort只返回索引
        ids_shuffle = torch.argsort(noise, dim=1)   # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep].to(self.device)  # ids_keep: [bs x len_keep x nvars]
        x_kept = torch.gather(s2p, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1,
                                                                              D))  # 按patch获取。gathet: https://zhuanlan.zhihu.com/p/352877584  # x_kept: [bs x len_keep x nvars  x patch_len]
        # start
        ori_kept = torch.gather(ori_s2q, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        # end
        # removed x
        x_removed = torch.zeros(bs, L - len_keep, D).to(self.device)   # [64,17,7,12]# x_removed: [bs x (L-len_keep) x nvars x patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len] [64,42,7,12]
        # start
        ori_k = torch.cat([ori_kept, x_removed], dim=1)
        # end
        # combine the kept part and the removed one,
        unf_masked = torch.gather(x_, dim=1,
                                  index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                                         D))  # x_masked: [bs x num_patch x nvars x patch_len]

        ori_masked = torch.gather(ori_k, dim=1,
                                  index=ids_restore.unsqueeze(-1).repeat(1, 1,
                                                                         D))  # x_masked: [bs x num_patch x nvars x patch_len]

        # fold = nn.Fold(output_size=(H, W), kernel_size=self.kernel, stride=self.kernel)
        f_masked = unf_masked.permute(0, 2, 1)
        x_enc_masked = self.fold(f_masked).squeeze(1)

        # start
        ori_masked = ori_masked.permute(0, 2, 1)
        ori_enc_masked = self.fold(ori_masked).squeeze(1)
        # end

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L]).to(self.device)   # (64,42,7)  # mask: [bs x num_patch x nvars]
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask_ush = torch.gather(mask, dim=1, index=ids_restore)  # mask对应x_masked.
        mask_ = mask_ush.unsqueeze(-1).expand_as(unf_masked).permute(0, 2, 1)
        mask_index = self.fold(mask_).squeeze(1)

        return x_enc_masked,ori_enc_masked, mask_index  # The input data has been masked; The masked data index

    def create_group(self, x_enc_masked, patch_len, stride):
        """
        xb: [bs x seq_len x n_vars]
        """
        seq_len = x_enc_masked.shape[1]
        num_patch = self.num_group
        tgt_len = patch_len + stride * (num_patch - 1)
        s_begin = seq_len - tgt_len

        xb = x_enc_masked[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]真正参与计算的矩阵，这也是目标值y
        x_patch_masked = xb.unfold(dimension=1, size=patch_len,
                                   step=stride)  # xb: [bs x num_patch x n_vars x patch_len]
        return x_patch_masked, num_patch, xb

    def create_blocks(self, x_enc_masked, patch_len, stride):
        """
        xb: [bs x seq_len x n_vars]
        """
        seq_len = x_enc_masked.shape[1]
        num_patch = self.num_group
        tgt_len = patch_len + stride * (num_patch - 1)
        s_begin = seq_len - tgt_len

        xb = x_enc_masked[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]真正参与计算的矩阵，这也是目标值y
        x_patch_masked = torch.stack(torch.split(xb, patch_len, dim=1),dim=1)
        return x_patch_masked, num_patch, xb

    def random_masking(self, xb, mask_ratio):
        # xb: [bs x num_patch x n_vars x patch_len]
        bs, L, nvars, D = xb.shape
        x = xb.clone()

        len_keep = int(L * (1 - mask_ratio))  # 每次mask是以patch为单位，保留的数量

        noise = torch.rand(bs, L, nvars, device=xb.device)  # noise in [0, 1], bs x L x nvars

        # sort noise for each sample, argsort只返回索引
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)  # ids_restore: [bs x L x nvars]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep, :]  # ids_keep: [bs x len_keep x nvars]
        x_kept = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, 1,
                                                                            D))  # 按patch获取。gathet: https://zhuanlan.zhihu.com/p/352877584  # x_kept: [bs x len_keep x nvars  x patch_len]

        # removed x
        x_removed = torch.zeros(bs, L - len_keep, nvars, D,
                                device=xb.device)  # [64,17,7,12]# x_removed: [bs x (L-len_keep) x nvars x patch_len]
        x_ = torch.cat([x_kept, x_removed], dim=1)  # x_: [bs x L x nvars x patch_len] [64,42,7,12]

        # combine the kept part and the removed one,
        x_masked = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, 1,
                                                                                  D))  # x_masked: [bs x num_patch x nvars x patch_len]

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([bs, L, nvars], device=x.device)  # (64,42,7)  # mask: [bs x num_patch x nvars]
        mask[:, :len_keep, :] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1,
                            index=ids_restore)  # mask对应x_masked.                              # [bs x num_patch x nvars]

        mask = mask.bool()  # mask: [bs x num_patch x n_vars]
        return x_masked, x_kept, mask, ids_restore



    def generate_blocks(self, x_enc_masked, patch_len, stride):
        """
        1： split blocks
        xb: [bs x seq_len x n_vars]
        """
        seq_len = x_enc_masked.shape[1]
        tgt_len = patch_len + stride * (self.num_group - 1)
        s_begin = seq_len - tgt_len

        xb = x_enc_masked[:, s_begin:, :]  # xb: [bs x tgt_len x nvars]真正参与计算的矩阵，这也是目标值y
        x_block = xb.unfold(dimension=1, size=patch_len, step=stride).transpose(-1,-2)  # xb: [bs x num_patch x n_vars x patch_len]
        return x_block

    def block_masking(self, x_block, mask_ratio):
        # 2 Random masking (32,60,16,8)
        mask_idx = []
        for i in range(self.num_group):
            mask_matrix = self.generate_mask(self.patch_size, self.args.enc_in, mask_ratio)
            mask_idx.append(mask_matrix)

        idx = torch.stack(mask_idx, dim=0)
        mask_index = idx.unsqueeze(0).expand_as(x_block).to(self.device)
        x_enc_masked = x_block * mask_index
        block_masked_index = (~mask_index.bool()).int() # 被掩码掉的位置置为1
        return x_enc_masked, block_masked_index  # The input data has been masked; The masked data index

    def generate_mask(self, rows, cols, ratio):
        matrix = np.zeros((rows, cols), dtype=int)

        # 为每行至少添加一个1
        for i in range(rows):
            rand_col = np.random.choice(cols, size=int(cols * (1 - ratio)), replace=False)
            matrix[i, rand_col] = 1

        # 确保每列至少有一个1
        for j in range(cols):
            if np.sum(matrix[:, j]) == 0:
                rand_row = np.random.randint(rows)
                matrix[rand_row, j] = 1

        return torch.from_numpy(matrix)



    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_ori, batch_y_ori, ori_data) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)

                ori_data = ori_data.float().to(self.device)
                # x_enc_masked, ori_enc_masked, mask_index = self.patch_Sequence(batch_x, ori_data,
                #                                                                        self.args.mask_rate)
                # x_patch_masked, num_patch, true_target = self.create_group(x_enc_masked,
                #                                                                    patch_len=self.args.patch_len,
                #                                                                    stride=self.args.patch_len)

                # start
                split_blocks = self.generate_blocks(batch_x, patch_len=self.args.patch_len, stride=self.args.patch_len)
                x_patch_masked, block_masked_index = self.block_masking(split_blocks, self.args.mask_rate)
                # end

                outputs = self.model(x_patch_masked,0,1)


                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()
                b, s, c = batch_x.shape
                mask_index = block_masked_index.reshape(b, -1, c)
                loss = self.patch_loss(pred, true, mask_index)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            # gpu_tracker.track()
            self.model.train()
            # gpu_tracker.track()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_ori, batch_y_ori, ori_data) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                ori_data =ori_data.float().to(self.device)
                # masking--patch  == ablation
                # x_enc_masked, _, mask_index = self.patch_Sequence(batch_x, ori_data,
                #                                                                self.args.mask_rate)
                # x_patch_masked, _,_ = self.create_group(x_enc_masked,
                #                                                            patch_len=self.args.patch_len,
                #                                                            stride=self.args.patch_len)

                # start
                split_blocks = self.generate_blocks(batch_x, patch_len=self.args.patch_len, stride=self.args.patch_len)
                x_patch_masked, block_masked_index = self.block_masking(split_blocks, self.args.mask_rate)
                # end
                outputs = self.model(x_patch_masked, epoch, i)  # (32,32,322,16)



                mask_index = block_masked_index.reshape(self.args.batch_size, -1, self.args.enc_in)
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # loss = criterion(outputs, batch_x)
                loss = self.patch_loss(outputs, batch_x, mask_index)
                # loss = self.patch_loss(outputs, split_blocks, block_masked_index)

                # end
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # best_model_path = path + '/' + 'checkpoint.pth'
        # self.model.load_state_dict(torch.load(best_model_path))

        path_e = self.model_path + '/' + self.model_name+'.pth'
        # path_e = self.model_path + '/' + setting + '/' + self.model_name+'.pth'
        torch.save(self.model, path_e)
        # print(path_e)
        return self.model

    def test(self, setting, test=0):

        print("EXP_long_term_forcast")
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # 预测
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = torch.zeros(10000)
        num = 1
        input_data = 0

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_ori, batch_y_ori, ori_data) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)

                ori_data = ori_data.float().to(self.device)

                # x_enc_masked, ori_enc_masked, mask_index = self.patch_Sequence(batch_x, ori_data,
                #                                                                        self.args.mask_rate)
                # x_patch_masked, num_patch, true_target = self.create_group(x_enc_masked,
                #                                                                    patch_len=self.args.patch_len,
                #                                                                    stride=self.args.patch_len)
                # start
                split_blocks = self.generate_blocks(batch_x, patch_len=self.args.patch_len, stride=self.args.patch_len)
                x_patch_masked, block_masked_index = self.block_masking(split_blocks, self.args.mask_rate)
                # end

                starter.record()
                outputs = self.model(x_patch_masked,0,1)
                # print(torch.cuda.memory_summary())
                # gpu_tracker.track()
                ender.record()
                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)  # 计算时间
                times[i] = curr_time
                num = i
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_x.detach().cpu().numpy()
                # batch_y = split_blocks.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                # if (1+i) % 5 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))
                    # break
                input_data = x_patch_masked

        flops, params = profile(self.model, inputs=(input_data, 0,1))
        print("Model Input Size:", input_data.shape)
        print("模型的复杂度FLOPs:", flops / 1e9, "模型的参数量：", params / 1000)
        inference_time = times[:num].mean().item()
        print("Inference Time:", inference_time)
        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        '''
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}'.format(mse, mae))
        f.write('\n')
        f.write('\n')
        f.close()
        '''

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        return

    def model_forecasting(self, setting):

        seq_x_path = r'F:\work\ai_work\AutoFormer\Time-Series-Library-main\dataset\seq_x.csv'
        seq_y_path = r'F:\work\ai_work\AutoFormer\Time-Series-Library-main\dataset\seq_y.csv'
        data_x = pd.read_csv(seq_x_path)
        data_y = pd.read_csv(seq_y_path)

        batch_x = torch.from_numpy(np.array(data_x.iloc[:,1:])).unsqueeze(0)
        batch_y = torch.from_numpy(np.array(data_y.iloc[:,1:])).unsqueeze(0)
        batch_x_ori = get_stamp(data_x[['date']])
        s, p = batch_x_ori.shape
        batch_x_ori = torch.from_numpy(batch_x_ori.reshape(1, s, p))
        batch_y_ori = get_stamp(data_y[['date']])
        s2, p2 = batch_y_ori.shape
        batch_y_ori = torch.from_numpy(batch_y_ori.reshape(1, s2, p2))

        self.model.eval()
        with torch.no_grad():
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)

            batch_x_ori = batch_x_ori.float().to(self.device)
            batch_y_ori = batch_y_ori.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder

            outputs, mask_index = self.model(batch_x, batch_x_ori, dec_inp, batch_y_ori)
            pred = outputs.detach().cpu().numpy()  # .squeeze()
            pred = pred[:, :, -1][0]

        df = pd.DataFrame({'preds': pred.tolist()})
        des_path = os.path.join(r'/Users/qiyuehan/Desktop/thirdPaper/code/PatchForecasting/my_models/result/TimesNet_pre_true.csv')
        df.to_csv(des_path, index=False, sep=',')
        return


