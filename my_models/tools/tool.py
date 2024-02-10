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

# from models.plot_heat import plot_heatmap
# from modeDls.wavelet.walvelet_without import wavelet_noising_column
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
                # assert w1/w2==2  # 预测长度比输入长度只能短1/2.
                repeat_num = self.weight.shape[-1]//self.w.shape[-1]
                add_w = self.w.repeat(1, repeat_num)   # 扩展维度1
                # add_w = torch.cat([self.w, self.w],dim=-1)
                weight = torch.cat([self.weight, add_w], dim=0)
            bias = torch.cat([self.bias, self.b], dim=0)
            bias = torch.tensor(bias, dtype=torch.float)
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight, self.bias)

class BModel(nn.Module):
    def __init__(self, in_dim, n_hidden):
        super(BModel, self).__init__()
        self.linear = MaskedLinear(in_dim, n_hidden, bias = True)
        self.sig = nn.Sigmoid()

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.linear(x)
        x = self.sig(x)
        return x

    def update_weights(self, w,b):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear.set_wb(w,b)

class blinear(nn.Module):
    def __init__(self, in_dim, n_hidden):
        super(blinear, self).__init__()
        self.linear = nn.Linear(in_dim, n_hidden, bias = True)
        self.sig = nn.Sigmoid()

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.linear(x)
        x = self.sig(x)
        return x

class update_elm(nn.Module):
    def __init__(self, in_dim, n_hidden):
        super(update_elm, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_dim, n_hidden, bias=True),
            # nn.Sigmoid(),
        )

    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.model(x)
        return x

def getDataMatrix(sequence, multi_uni="M", seq_len=96,  pre_len=48, stride = 1):

    if multi_uni=="MS":
        data_len = len(sequence) - seq_len - pre_len
        X = np.zeros(shape=(data_len, seq_len, sequence.shape[1]))
        T = np.zeros(shape=(data_len, pre_len))
        for i in range(data_len):
            X[i, :] = np.array(sequence[i:(i + seq_len)])
            T[i, :] = np.array(sequence.iloc[:, -1][(i + seq_len):(i + seq_len + pre_len)])
    else:
        # 多维度 10   5
        data_len = int((len(sequence)-seq_len-pre_len)/stride)-1
        # data_len = len(sequence)-seq_len-pre_len
        X = np.zeros(shape=(data_len, seq_len, sequence.shape[1]))
        T = np.zeros(shape=(data_len, pre_len, sequence.shape[1]))
        j = 0
        for i in range(data_len):
            X[i, :] = np.array(sequence[j:(j + seq_len)])
            T[i, :] = np.array(sequence[(j + seq_len):(j + seq_len + pre_len)])
            j = j + stride

    print("多维度-多维度！")
    return X, T

# 原来的cov_matrix
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

# second modify
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_period(nn.Module):
    """
    Series period+噪声
    """
    def __init__(self, kernel_size):
        super(series_period, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res

# multivariate relationships
def cov_matrix2(input, n_dim):
    kernel_size = 25
    series_p = series_period(kernel_size)
    x = series_p(input)
    features_rela = torch.matmul(x.T, x) # 空间非平稳信息；
    sample_tril = torch.tril(features_rela, diagonal=-1)
    sample_triu = torch.triu(features_rela, diagonal=1)
    sample_sum = sample_tril + sample_triu
    cov_matrix = torch.softmax(sample_sum / torch.sqrt(torch.tensor(n_dim)), dim=1)
    return cov_matrix


def get_relations(input, dmodel=512):
    mean_in = torch.mean(input, dim=0)
    mean_in = mean_in.expand(input.size())
    x = input - mean_in
    x_T = x.permute(0, 2, 1)
    sample_matrix = torch.matmul(x, x_T)

    sample_tril = torch.tril(sample_matrix, diagonal=-1)
    sample_triu = torch.triu(sample_matrix, diagonal=1)
    sample_sum = sample_tril + sample_triu
    sample_cor = torch.matmul(torch.softmax(sample_sum/torch.sqrt(torch.tensor(dmodel)), dim=1), input)

    features_rela = torch.matmul(x_T, x)
    features_tril = torch.tril(features_rela, diagonal=-1)
    features_triu = torch.triu(features_rela, diagonal=1)
    features_sum = features_tril + features_triu
    # plot_heatmap(torch.softmax(features_rela, dim=0))
    cov_matrix = torch.softmax(features_sum / torch.sqrt(torch.tensor(dmodel)), dim=1)
    features_cor = torch.matmul(input, cov_matrix)
    return sample_cor, None, features_cor


##  Ablation
def get_correlations(x_input, multi_f, num):

    # x_input = torch.from_numpy(x_input)
    # multi_f:不同维度之间的关联关系
    X_multi = torch.matmul(x_input, multi_f)
    # 长时间序列长期相关性关系
    x_input_T = x_input.permute(0, 2, 1)
    X_rela = torch.matmul(x_input, x_input_T)
    X_relations = torch.matmul(X_rela, x_input)

    # 完整的ori
    input = X_multi + X_relations

    # (2)：without sequence feature extraction
    # input  = X_multi

    # (3): without varialbe correlation extraction
    # input = X_relations

    input = input.float().reshape(num, -1)
    return input

def plot_res(true_data, y_pred, pre_len, segment, filename):
    td = list(true_data[0])
    pd = list(y_pred[0])
    for i in range(0, segment):
        # true_datas = td + list(true_data[i])
        # pre_datas = pd + list(y_pred[i])
        true_datas = td
        pre_datas = pd
        td = true_datas
        pd = pre_datas
    # plt.figure(figsize=(8, 4))
    # plt.plot(true_datas, c='b', label="GroundTruth")  # 测试数组
    # plt.plot(pre_datas, c='orange', label='Prediction ')  # 测试数组
    # plt.title("_"+filename)  # 标题

    plt.plot(true_datas, label="GroundTruth")  # 测试数组
    plt.plot(pre_datas,  label='Prediction ')  # 测试数组

    plt.xlabel('Time', fontproperties='Times New Roman', fontsize=20)
    plt.ylabel('Value', fontproperties='Times New Roman', fontsize=20)
    plt.legend()
    plt.savefig(r'F:\work\ai_work\zd\pyoselm-master\pytorch_project\result\ablation.pdf',dpi=500)
    plt.show()


# def waveNoising(sequence,ndim):
#     # 训练数据使用小波去噪，测试数据集不需要
#     df = wavelet_noising(sequence, ndim)
#     df = df.T
#     return df


def norm_data(sequence):
    scaler = StandardScaler()  # 标准化转换
    scaler.fit(sequence)  # 训练标准化对象
    seq = scaler.transform(sequence)  # 转换数据集
    return seq

def get_step(end_stop):
    if end_stop >= 10000:
        step = 100
    elif end_stop >= 2000:
        step = 50
    else:
        step = 10
    return step


def min_mse(in_dim, x_train, y_train, x_test, y_test, start, stop, step):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    get_mse_mean = []
    for n_hidden in range(start, stop, step):
        net = ELM(in_dim, n_hidden).to(device)
        outputs = net(x_train)
        # 求伪逆  H beta = T   -> beta = H'T  matmul:多维矩阵相乘  (未使用正则化)
        beta = torch.matmul(torch.pinverse(outputs), y_train)
        # 预测
        with torch.no_grad():
            # 得到H
            H = net(x_test)
            predict = torch.matmul(H, beta)

            # 计算测试精度
            MSE = mean_squared_error(y_test, predict.cpu())
            MAE = mean_absolute_error(y_test, predict.cpu())
            get_mse_mean.append(MSE)
    res_mse = np.min(get_mse_mean)
    # res_mse = (np.sum(get_mse_mean) - np.max(get_mse_mean))/(len(get_mse_mean)-1)
    print(str(n_hidden)+"_MSE:" + str(MSE)+",_MAE:" + str(MAE))
    return res_mse

def get_interval(in_dim, x_train, y_train, x_test, y_test, start, stop):
    end_stop = stop - start
    mid = end_stop // 2 + start
    mid_div = mid
    while mid_div >= 100:
        step = get_step(end_stop)
        res_mse1 = min_mse(in_dim, x_train, y_train, x_test, y_test, start, mid, step)
        res_mse2 = min_mse(in_dim, x_train, y_train, x_test, y_test, mid, stop, step)
        if res_mse1 > res_mse2:
            start = mid
            stop = stop
            end_stop = stop - start
        else:
            start = start
            stop = mid
            end_stop = stop - start
        mid_div = end_stop // 2
        mid = start + mid_div
    print("Start:", start, "Stop:", stop)
    return start, stop


def plot_factors(file_name):
    data = pd.read_csv(file_name)
    num = data['hidden']
    mse = data['mse']
    R2 = data['R2']
    mae = data['mae']
    cost_time = data['time']

    plt.figure(figsize=(10, 5))
    plt.plot(num, mse, 'r--', label='MSE')
    # plt.plot(num, mae, 'b-.',  label='MAE')
    # plt.plot(num, R2, 'g-*', label='R2')
    # plt.plot(num, cost_time, 'y-+', label='Time')
    plt.xlabel("Number of hidden layer nodes", fontproperties='Times New Roman', fontsize=20)
    plt.ylabel("Value", fontproperties='Times New Roman', fontsize=20)
    plt.legend()
    plt.savefig(r'F:\work\ai_work\zd\pyoselm-master\pytorch_project\result\nodes.pdf')
    plt.show()


# 模型的大小
def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型参数量：', param_sum)
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)


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

    # 深度可分离卷积
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