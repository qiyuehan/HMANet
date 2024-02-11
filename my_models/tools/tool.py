import numpy as np
import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, \
    r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

import seaborn as sns
import torch.nn.functional as F
np.random.seed(0)
torch.manual_seed(0)

def to_var(x, requires_grad=False):
    """
    Automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x.clone().detach().requires_grad_(requires_grad)


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.up_wb = False
        self.mask = None

    def set_wb(self, w,b):
        self.w = to_var(w, requires_grad=False)
        self.b = to_var(b, requires_grad=False)
        # self.weight.data = self.weight.data * self.mask.data
        self.up_wb = True

    def get_mask(self):
        print(self.mask_flag)
        return self.mask

    def forward(self, x):
        if self.up_wb:
            _,w1 = self.weight.shape
            _,w2 = self.w.shape
            if w1 <= w2:
                weight = torch.cat([self.weight, self.w[:,:self.weight.shape[-1]]],dim=0)
            else:
               
                repeat_num = self.weight.shape[-1]//self.w.shape[-1]
                add_w = self.w.repeat(1, repeat_num)  
                # add_w = torch.cat([self.w, self.w],dim=-1)
                weight = torch.cat([self.weight, add_w], dim=0)
            bias = torch.cat([self.bias, self.b], dim=0)
            bias = torch.tensor(bias, dtype=torch.float)
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight, self.bias)




def getDataMatrix(sequence, multi_uni="M", seq_len=96,  pre_len=48, stride = 1):

    if multi_uni=="MS":
        data_len = len(sequence) - seq_len - pre_len
        X = np.zeros(shape=(data_len, seq_len, sequence.shape[1]))
        T = np.zeros(shape=(data_len, pre_len))
        for i in range(data_len):
            X[i, :] = np.array(sequence[i:(i + seq_len)])
            T[i, :] = np.array(sequence.iloc[:, -1][(i + seq_len):(i + seq_len + pre_len)])
    else:
       
        data_len = int((len(sequence)-seq_len-pre_len)/stride)-1
        # data_len = len(sequence)-seq_len-pre_len
        X = np.zeros(shape=(data_len, seq_len, sequence.shape[1]))
        T = np.zeros(shape=(data_len, pre_len, sequence.shape[1]))
        j = 0
        for i in range(data_len):
            X[i, :] = np.array(sequence[j:(j + seq_len)])
            T[i, :] = np.array(sequence[(j + seq_len):(j + seq_len + pre_len)])
            j = j + stride


    return X, T


def cov_matrix_ori(input, n_dim):
    mean_in = torch.mean(input, dim=0)
    mean_in = mean_in.expand(input.size())
    x = input - mean_in
    features_rela = torch.matmul(x.T, x)
    cov_matrix = torch.softmax(features_rela / torch.sqrt(torch.tensor(n_dim)), dim=1)
    return cov_matrix

def cov_matrix(input, n_dim):
    mean_in = torch.mean(input, dim=0)
    mean_in = mean_in.expand(input.size())
    x = input - mean_in
    features_rela = torch.matmul(x.T, x)
    sample_tril = torch.tril(features_rela, diagonal=-1)
    sample_triu = torch.triu(features_rela, diagonal=1)
    sample_sum = sample_tril + sample_triu
    cov_matrix = torch.softmax(sample_sum / torch.sqrt(torch.tensor(n_dim)), dim=1)
    return cov_matrix


def norm_data(sequence):
    scaler = StandardScaler()  
    scaler.fit(sequence)  
    seq = scaler.transform(sequence) 
    return seq



def smape_score(y_true, y_pred):
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))


def mul_factors(true_data, predict):
    mae = mean_absolute_error(true_data, predict)
    mse = mean_squared_error(true_data, predict)
    rmse = np.sqrt(mean_squared_error(true_data, predict))
    mape = mean_absolute_percentage_error(true_data, predict)
    smape =smape_score(true_data, predict)
    r2 = r2_score(true_data, predict)

    return mae, mse, rmse, mape, smape, r2

class modify_deep(nn.Module):
    def __init__(self,in_size, out_size,ker_size=3,stride=1,padding=1,dilation=0):
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

    def forward(self,x):
        x1 = self.depth_conv(x)
        x2 = self.point_conv(x1)
        x3 = self.relu(x2)
        x4 = self.bn(x3)
        return x4


class squeeze_series(nn.Module):
    def __init__(self, wave='haar', J=1, device='cpu'):
        super(squeeze_series, self).__init__()
        self.dwt = DWT1DForward(wave=wave, J=J).to(device)
    def forward(self, input):
        input = torch.tensor(input.transpose(-1, -2), dtype=torch.float)
        yl, yh = self.dwt(input)
        return yl, yh[0]



if __name__ == '__main__':
    file = r'F:\work\ai_work\zd\pyoselm-master\result\res.csv'
    plot_factors(file)
