#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

data = np.loadtxt('data6.txt', delimiter='\t')  # ��ȡ����
print(data.shape)
# Z-Score��׼��
zscore_scaler = preprocessing.StandardScaler()  # ����StandardScaler����
data_scale_1 = zscore_scaler.fit_transform(data)  # StandardScaler��׼������

# Max-Min��׼��
minmax_scaler = preprocessing.MinMaxScaler()  # ����MinMaxScalerģ�Ͷ���
data_scale_2 = minmax_scaler.fit_transform(data)  # MinMaxScaler��׼������

# MaxAbsScaler��׼��
maxabsscaler_scaler = preprocessing.MaxAbsScaler()  # ����MaxAbsScaler����
data_scale_3 = maxabsscaler_scaler.fit_transform(data)  # MaxAbsScaler��׼������

# RobustScaler��׼��
robustscalerr_scaler = preprocessing.RobustScaler()  # ����RobustScaler��׼������
data_scale_4 = robustscalerr_scaler.fit_transform(data)  # RobustScaler��׼����׼������

# չʾ��������
data_list = [data, data_scale_1, data_scale_2, data_scale_3,
             data_scale_4]  # �������ݼ��б�
scalar_list = [15, 10, 15, 10, 15, 10]  # ������ߴ��б�
color_list = ['black', 'green', 'blue', 'yellow', 'red']  # ������ɫ�б�
merker_list = ['o', ',', '+', 's', 'p']  # ������ʽ�б�
title_list = ['source data', 'zscore_scaler', 'minmax_scaler',
              'maxabsscaler_scaler', 'robustscalerr_scaler']  # ���������б�
for i, data_single in enumerate(data_list):  # ѭ���õ�������ÿ����ֵ
    plt.subplot(2, 3, i+1)  # ȷ��������   �������У��У�ͼ��������
    plt.scatter(data_single[:, :-1], data_single[:, -1], s=scalar_list[i],
                marker=merker_list[i],c=color_list[i])  # ������չʾɢ��ͼ
    plt.title(title_list[i])  # �������������
plt.suptitle("raw data and standardized data")  # �����ܱ���
plt.show()  # չʾͼ��
