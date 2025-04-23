import argparse
import torch
import random
import numpy as np
from my_models.utils import Block_Rep
fix_seed = 2024 # 2023 2025
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
parser = argparse.ArgumentParser(description='HMANet')
parser.add_argument('--model_id', type=str, required=False, default='ETTh1', help='Model id')
parser.add_argument('--model', type=str, required=False, default='HMANet')

# data loader
parser.add_argument('--data', type=str, required=False, default='custom', help='dataset type')
parser.add_argument('--root_path', type=str, default=r'C:\Users\11848\Desktop\TNNLS P2\HMANet_Upload\dataset\\', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--checkpoints', type=str, default=r'C:\Users\11848\Desktop\TNNLS P2\HMANet_Upload\Pre_Train\PreTrain_checkpoint\\', help='location of model checkpoints')

# forecasting task
parser.add_argument('--pre_train', type=int, default=512, help='input sequence length')
parser.add_argument('--patch_len', type=int, default=16, help='start token length')
parser.add_argument('--label_len', type=int, default=0, help='start token length')
parser.add_argument('--pred_len', type=int, default=0, help='prediction sequence length')
parser.add_argument('--dilation_rate', type=list, default=[1,2,5],help='HDC')
parser.add_argument('--mask_rate', type=float, default=0.6, help='mask ratio')
parser.add_argument('--kernel_size', type=tuple, default=(8, 2), help='kernel')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
parser.add_argument('--train_percent', type=float, default=0.5, help='train data rate')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--use_gpu', type=str, default='True')
parser.add_argument('--devices', type=str, default='cuda', help='device ids of multile gpus')

if __name__ == '__main__':
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    print('Args in experiment:')
    print(args)
    Exp = Block_Rep
    for i in range(args.itr):
        setting = '{}_{}_{}'.format(args.model_id, args.data, i)
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()

