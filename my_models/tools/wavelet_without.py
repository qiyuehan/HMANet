import numpy as np
import math
import pywt

def sgn(num):
    if(num > 0.0):
        return 1.0
    elif(num == 0.0):
        return 0.0
    else:
        return -1.0

def wavelet_noising_uni(new_df):
    data = new_df.values.T.tolist()
    w = pywt.Wavelet('sym8')
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)
    length1 = len(cd1)
    length0 = len(data)
    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)
    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = []
    usecoeffs.append(ca5)
    a = 0.5
    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k].any()) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs

def wavelet_noising_column(new_df):
    res = np.zeros_like(new_df)
    row_num, colum_num = new_df.shape
    for i in range(colum_num):
        data = new_df.iloc[:, i]
        data = data.T.tolist()
        w = pywt.Wavelet('sym8')
        [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)

        length1 = len(cd1)
        length0 = len(data)

        Cd1 = np.array(cd1)
        abs_cd1 = np.abs(Cd1)
        median_cd1 = np.median(abs_cd1)

        sigma = (1.0 / 0.6745) * median_cd1
        lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
        usecoeffs = []
        usecoeffs.append(ca5)
        a = 0.5

        for k in range(length1):
            if (abs(cd1[k]) >= lamda):
                cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
            else:
                cd1[k] = 0.0

        length2 = len(cd2)
        for k in range(length2):
            if (abs(cd2[k]) >= lamda):
                cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
            else:
                cd2[k] = 0.0

        length3 = len(cd3)
        for k in range(length3):
            if (abs(cd3[k]) >= lamda):
                cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
            else:
                cd3[k] = 0.0

        length4 = len(cd4)
        for k in range(length4):
            if (abs(cd4[k]) >= lamda):
                cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
            else:
                cd4[k] = 0.0

        length5 = len(cd5)
        for k in range(length5):
            if (abs(cd5[k]) >= lamda):
                cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
            else:
                cd5[k] = 0.0

        usecoeffs.append(cd5)
        usecoeffs.append(cd4)
        usecoeffs.append(cd3)
        usecoeffs.append(cd2)
        usecoeffs.append(cd1)
        recoeffs = pywt.waverec(usecoeffs, w)
        res[:, i] = recoeffs[:row_num]
    return res