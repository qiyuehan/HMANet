import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
from my_models.tools.tool import norm_data, getDataMatrix
from my_models.tools.wavelet_without import wavelet_noising_column
from utils.SLFN import SLFNet
import warnings
warnings.filterwarnings("ignore")

def generate_blocks(num_group, x_enc_masked, patch_len, stride):
    seq_len = x_enc_masked.shape[1]
    tgt_len = patch_len + stride * (num_group - 1)
    s_begin = seq_len - tgt_len
    xb = x_enc_masked[:, s_begin:, :]
    x_block = xb.unfold(dimension=1, size=patch_len, step=stride).transpose(-1,-2)
    return x_block


def main():
    np.random.seed(0)
    torch.manual_seed(0)
    parser = argparse.ArgumentParser(description=' for Time Series Forecasting')
    parser.add_argument('--root_path', type=str,
                        default=r'.\dataset\', help='root path of the data file')
    parser.add_argument('--data', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--PreTrain_checkpoint', type=str,
                        default=r'.\Pre_Train\PreTrain_checkpoint\',
                        help='model path')
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length=seq_len-pred_len')
    parser.add_argument('--pre_train', type=int, default=512, help='input sequence length=seq_len-pred_len')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--fea_dim', type=int, default=7, help='output dim')
    parser.add_argument('--epochs', type=int, default=20, help='epoch')
    parser.add_argument('--patch_size', type=int, default=16, help='patch_len')
    parser.add_argument('--add_nodes', type=int, default=5, help='add hidden nodes per')
    parser.add_argument('--train_percent', type=float, default=0.7, help='split train data and val data')
    parser.add_argument('--test_percent', type=float, default=0.35, help='split train data and val data')
    parser.add_argument('--patience', type=int, default=5, help='split train data and val data')
    parser.add_argument('--stride', type=int, default=1, help='split train data')
    parser.add_argument('--features', type=str, default='M', help='split train data')
    parser.add_argument('--model_name', type=str, default='ETTh1', help='data model name')
    parser.add_argument('--ext_nodes', type=int, default=5, help='enhance nodes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--n_hidden_dim', type=int, default=1000, help='the number of hidden nodes')
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(args.device)

    filePath = args.root_path + args.data
    df_ori = pd.read_csv(filePath, header=0, encoding='latin-1')
    data_start = int(len(df_ori) * args.train_percent)
    print("Dataset Length:", int(len(df_ori)) - data_start)
    df_ori = df_ori[data_start:]
    print("Test_set Length:", int(len(df_ori)))
    seq_in = df_ori.iloc[:, 1:].astype(float)
    _, nf = seq_in.shape
    df_wave = wavelet_noising_column(seq_in)
    seq_wave = norm_data(df_wave)
    X_wave, Y_wave = getDataMatrix(seq_wave,args.features, args.seq_len, args.pred_len, args.stride)
    train_data, test_data, _, _ = train_test_split(X_wave, Y_wave, test_size=args.test_percent, random_state=1)
    num_group = (max(args.pre_train, args.patch_size) - args.patch_size) // args.patch_size + 1
    seq_in = np.nan_to_num(seq_in)
    norm_ori = norm_data(seq_in)
    X_norm, Y_norm = getDataMatrix(norm_ori,args.features, args.seq_len, args.pred_len, args.stride)
    _, _, train_target, test_target = train_test_split(X_norm, Y_norm, test_size=args.test_percent, random_state=1)
    total_train, s, l = train_data.shape
    in_dim = s * l
    t_train = torch.from_numpy(train_data).to(args.device)
    t_train = t_train.nan_to_num(0.0)
    t_test = torch.from_numpy(test_data).to(args.device)
    t_test = t_test.nan_to_num(0.0)
    in_train = torch.tensor(t_train, dtype=torch.float)
    tar_train = torch.from_numpy(train_target)
    in_test = torch.tensor(t_test, dtype=torch.float)
    tar_test = torch.from_numpy(test_target).float()
    model_path = args.PreTrain_checkpoint + args.model_name + '.pth'
    model_pre = torch.load(model_path, map_location=args.device)
    for param in model_pre.parameters():
        param.requires_grad = False
    train_block = torch.split(in_train, args.batch_size, dim=0)
    batch_train = torch.stack(train_block[:-1], dim=0)
    batch_tar_train = torch.stack(torch.split(tar_train, args.batch_size, dim=0)[:-1], dim=0)
    # Test
    test_block = torch.split(in_test, args.batch_size, dim=0)
    batch_test = torch.stack(test_block[:-1], dim=0)
    batch_tar_test = torch.stack(torch.split(tar_test, args.batch_size, dim=0)[:-1], dim=0)
    batch_train_ori = batch_train.float().to(args.device)
    batch_tar_train_ori = batch_tar_train.float().to(args.device)
    batch_test_ori = batch_test.float().to(args.device)
    batch_tar_test_ori = batch_tar_test.float().to(args.device)
    with torch.no_grad():
        net, b_beta = train_valid(args, model_pre, batch_train, batch_train_ori, batch_tar_train, batch_tar_train_ori,
                                  batch_tar_test, in_dim, num_group)
        print("*******" * 15)
        print("--------------------Test Running ------------------")
        test_Result(args, net, model_pre, b_beta, batch_test, batch_test_ori, batch_tar_test, batch_tar_test_ori,
                    num_group)


def get_representation(args, model_pre, batch_train, batch_train_ori, num_group):
    get_representation = []
    for i in range(len(batch_train)):
        batch_x = batch_train[i]
        b, _, c = batch_x.shape
        dec_inp_zeros = torch.zeros(b, args.pre_train -args.seq_len,c).float().to(args.device)
        dec_inp_pre_model = torch.cat([batch_x, dec_inp_zeros], dim=1).float().to(args.device)
        split_blocks = generate_blocks(num_group, dec_inp_pre_model, patch_len=args.patch_size, stride=args.patch_size)
        pre_fea = model_pre(split_blocks)
        get_representation.append(pre_fea)
    f_dim = 0
    pre_rep = torch.cat(get_representation, dim=0)
    train_rep = pre_rep[:, :args.seq_len, f_dim:]
    train_inputs = (batch_train_ori.reshape(-1, args.seq_len, args.fea_dim) + train_rep).reshape(
            train_rep.size(0), -1)
    return train_inputs


def test_representation(args, model_pre, batch_train, batch_train_ori, num_group):
    get_representation = []
    for i in range(len(batch_train)):
        batch_x = batch_train[i]
        b,_,c = batch_x.shape
        dec_inp_zeros = torch.zeros(b, args.pre_train -args.seq_len, c).float().to(args.device)
        dec_inp_pre_model = torch.cat([batch_x, dec_inp_zeros], dim=1).float().to(args.device)
        split_blocks = generate_blocks(num_group, dec_inp_pre_model, patch_len=args.patch_size, stride=args.patch_size)
        pre_fea = model_pre(split_blocks)
        get_representation.append(pre_fea)

    f_dim = 0
    pre_rep = torch.stack(get_representation, dim=0)
    train_rep = pre_rep[:, :, :args.seq_len, f_dim:]
    train_inputs = (batch_train_ori + train_rep)
    return train_inputs


def train_valid(args, model_pre, batch_train, batch_train_ori, batch_tar_train, batch_tar_train_ori, batch_tar_test,
                 in_dim, num_group):
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    device = args.device
    epoch_now = 0
    num_addnodes = args.add_nodes
    bmse = 1e9
    bmae = 1e7
    best_model = None
    b_beta = None
    criterion = nn.MSELoss()
    _, _, pl, c = batch_tar_test.shape
    while epoch_now < args.epochs:
        n_hidden_dim = args.n_hidden_dim + num_addnodes * epoch_now
        net = SLFNet(in_dim, n_hidden_dim).to(device)
        data_length = len(batch_train)
        train_len = int(0.8 * data_length)
        batch_train1, batch_test = batch_train[:train_len], batch_train[train_len:]
        batch_tar_train_ori1, batch_tar_test_ori = batch_tar_train_ori[:train_len], batch_tar_train_ori[train_len:]
        batch_train_ori1, batch_test_ori = batch_train_ori[:train_len], batch_train_ori[train_len:]
        _, _, p, v = batch_tar_train_ori.shape
        batch_tar_train_ori1 = batch_tar_train_ori1.reshape(-1, p * v)
        batch_tar_test_ori = batch_tar_test_ori.reshape(-1, p * v).cpu()
        train_inputs = get_representation(args, model_pre, batch_train1, batch_train_ori1,num_group)
        test_inputs = get_representation(args, model_pre, batch_test, batch_test_ori, num_group)
        beta = net.train(train_inputs, batch_tar_train_ori1)
        test_output = net.predict(test_inputs, beta)
        MSE = mean_squared_error(batch_tar_test_ori, test_output.cpu())
        MAE = mean_absolute_error(batch_tar_test_ori, test_output.cpu())
        loss = criterion(batch_tar_test_ori, test_output.cpu())
        if MSE < bmse:
            bmse = MSE
            bmae = MAE
            best_hidden = n_hidden_dim
            b_beta = beta
            best_model = net
            print("hidden_nodes:", best_hidden, "Average MSE:", bmse, "MAE:", bmae)
        epoch_now += 1
        early_stopping(loss, b_beta)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    print("Training Result-> Best MSE:", bmse, "Best MAE:", bmae)
    return best_model, b_beta


def test_Result(args, net, model_pre, b_beta, batch_test, batch_test_ori, batch_tar_test, batch_tar_test_ori,
                num_group):
    _, _, p, v = batch_tar_test_ori.shape
    test_mse = []
    test_mae = []
    test_loss = []
    criterion = nn.MSELoss()
    test_inputs = test_representation(args, model_pre, batch_test, batch_test_ori, num_group)
    for i in range(len(test_inputs)):
        b_test = test_inputs[i]
        target = batch_tar_test_ori[i].reshape(args.batch_size, -1).cpu()
        b_test = b_test.reshape(args.batch_size, -1)
        output = net.predict(b_test, b_beta)
        MSE = mean_squared_error(target, output.cpu())
        MAE = mean_absolute_error(target, output.cpu())
        test_mse.append(MSE)
        test_mae.append(MAE)
        loss = criterion(target, output.cpu())
        test_loss.append(loss)
    print("Test Result --> Mean_MSE:", np.mean(test_mse), "Mean_MAE:", np.mean(test_mae))
    f = open(r".\result_LSTF.txt", 'a')
    f.write(args.model_name +'_'+str(args.pred_len) + "  \n")
    f.write('mse:{}, mae:{}'.format(np.mean(test_mse), np.mean(test_mae)))
    f.write('\n')
    f.close()


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss

if __name__ == '__main__':
    main()
