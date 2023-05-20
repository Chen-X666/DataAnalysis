#!/usr/bin/env python
# coding:gbk,
# �����Ӫ���ݵĹ���������

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, LinearRegression
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt

# ��ȡ����
data = np.loadtxt('data5.txt', delimiter='\t')  # ��ȡ�����ļ�
print(data.shape)
x, y = data[:, :-1], data[:, -1]  # �з��Ա�����Ԥ�����
correlation_matrix = np.corrcoef(x, rowvar=0)  # ����Է���
print("���ϵ������\n %s" % correlation_matrix.round(2))  # ��ӡ�������Խ��

print('ʹ����ع��㷨���лع����')
model_ridge = Ridge(alpha=1.0)  # ������ع�ģ�Ͷ���
model_ridge.fit(x, y)  # ����x/yѵ��ģ��
print(model_ridge.coef_)  # ��ӡ����Ա�����ϵ��
print(model_ridge.intercept_)  # ��ӡ����ؾ�
print(model_ridge.score(x, y))  # R��

print('ʹ����ͨ�����Իع�')
m1 = LinearRegression()
m1.fit(x, y)
print(m1.score(x, y))

print('ʹ��rigdeCV��ȡ���alpha')
model = RidgeCV(alphas=[0.1, 1.0, 10.0], store_cv_values=True)  # ͨ��RidgeCV�������ö������ֵ���㷨ʹ�ý�����֤��ȡ��Ѳ���ֵ
model.fit(x, y)
print("Best alpha using built-in RidgeCV: %s" % model.alpha_)

model_ridge = Ridge(alpha=0.1)  # ������ع�ģ�Ͷ���
model_ridge.fit(x, y)  # ����x/yѵ��ģ��
print(model_ridge.score(x, y))  # R��

print('\n ʹ�����ɷֻع���лع����')
model_pca = PCA()  # ����PCAģ�Ͷ���
data_pca = model_pca.fit_transform(x)  # ��x�������ɷַ���
print(x[:2, :])
k = model_pca.inverse_transform(data_pca) # ��ά�������ת��Ϊԭʼ����
print(k[:2, :])
ratio_cumsm = np.cumsum(
    model_pca.explained_variance_ratio_)  # �õ��������ɷַ���ռ�ȵ��ۻ�����
print(ratio_cumsm)  # ��ӡ����������ɷַ���ռ���ۻ�
rule_index = np.where(ratio_cumsm > 0.8)  # ��ȡ����ռ�ȳ���0.8����������ֵ
print("rule_index:{}".format(rule_index))
min_index = rule_index[0][0]  # ��ȡ��С����ֵ
print("min_index:{}".format(min_index))
data_pca_result = data_pca[:, :min_index + 1]  # ������С����ֵ��ȡ���ɷ�
model_liner = LinearRegression()  # �����ع�ģ�Ͷ���
model_liner.fit(data_pca_result, y)  # �������ɷ����ݺ�Ԥ�����y��ѵ��ģ��
print("�Ա�����ϵ��:{}".format(model_liner.coef_))  # ��ӡ����Ա�����ϵ��
print(model_liner.intercept_)  # ��ӡ����ؾ�
print(model_liner.score(data_pca_result, y))



