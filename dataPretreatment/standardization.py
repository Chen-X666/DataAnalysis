#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def standardizationPic(data):
    data = np.array(data)
    print(data.shape)
    print(data)
    data_scale_1, data_scale_2, data_scale_3, data_scale_4 = ZScore(data), MaxMin(data), MaxAbsScaler(
        data), RobustScaler(data)
    data_list = [data, data_scale_1, data_scale_2, data_scale_3,
                 data_scale_4]
    scalar_list = [15, 10, 15, 10, 15, 10]
    color_list = ['black', 'green', 'blue', 'yellow', 'red']
    merker_list = ['o', ',', '+', 's', 'p']
    title_list = ['source data', 'zscore_scaler', 'minmax_scaler',
                  'maxabsscaler_scaler', 'robustscalerr_scaler']
    for i, data_single in enumerate(data_list):
        plt.subplot(2, 3, i + 1)  # ȷ��������
        plt.scatter(data_single[:, :-1], data_single[:, -1], s=scalar_list[i],
                    marker=merker_list[i], c=color_list[i])
        plt.title(title_list[i])
    plt.suptitle("raw data and standardized data")
    plt.show()  # չʾͼ��

def ZScore(data):
    zscore_scaler = preprocessing.StandardScaler()  # ����StandardScaler����
    data_scale_1 = zscore_scaler.fit_transform(data)  # StandardScaler��׼������
    return data_scale_1

def MaxMin(data):
    minmax_scaler = preprocessing.MinMaxScaler()  # ����MinMaxScalerģ�Ͷ���
    data_scale_2 = minmax_scaler.fit_transform(data)  # MinMaxScaler��׼������
    return data_scale_2

def MaxAbsScaler(data):
    maxabsscaler_scaler = preprocessing.MaxAbsScaler()  # ����MaxAbsScaler����
    data_scale_3 = maxabsscaler_scaler.fit_transform(data)  # MaxAbsScaler��׼������
    return data_scale_3

def RobustScaler(data):
    robustscalerr_scaler = preprocessing.RobustScaler()  # ����RobustScaler��׼������
    data_scale_4 = robustscalerr_scaler.fit_transform(data)  # RobustScaler��׼����׼������
    return data_scale_4

if __name__ == '__main__':
    print()