import numpy as np
from sklearn.preprocessing import StandardScaler
np.random.seed(0)
# seq_in, args.features, args.seq_len, args.pred_len, args.stride
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
        X = np.zeros(shape=(data_len, seq_len, sequence.shape[1]))
        T = np.zeros(shape=(data_len, pre_len, sequence.shape[1]))
        j = 0
        for i in range(data_len):
            X[i, :] = np.array(sequence[j:(j + seq_len)])
            T[i, :] = np.array(sequence[(j + seq_len):(j + seq_len + pre_len)])
            j = j + stride
    return X, T

def norm_data(sequence):
    scaler = StandardScaler()
    scaler.fit(sequence)
    seq = scaler.transform(sequence)
    return seq
