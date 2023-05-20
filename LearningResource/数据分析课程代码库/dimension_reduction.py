#!/usr/bin/env python
#coding:gbk,
# PCA��ά
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# ��ȡ�����ļ�
data = np.loadtxt('data1.txt')  # ��ȡ�ı������ļ�
x = data[:, :-1]  # ��������x
y = data[:, -1]  # ���Ŀ�����y
print(np.size(x,1))#��������ά��
print(np.size(x,0))
print(x[0], y[0])  # ��ӡ���x��y�ĵ�һ����¼

print('\nʹ��sklearn��DecisionTreeClassifier�жϱ�����Ҫ��')
model_tree = DecisionTreeClassifier(random_state=0)  # �������������ģ�Ͷ���
model_tree.fit(x, y)  # �����ݼ���ά�Ⱥ�Ŀ���������ģ��
feature_importance = model_tree.feature_importances_  # ������б�������Ҫ�Ե÷�
print('The importance score of each parameter:')
print(feature_importance)  # ��ӡ���

print('\nʹ��sklearn��PCA����ά��ת��')
model_pca = PCA()  # ����PCAģ�Ͷ���
#n_components=3
#n_components='mle'
model_pca.fit(x)  # �����ݼ�ѵ��PCAģ��
model_pca.transform(x)  #��Xת���ɽ�ά�������
components = model_pca.components_  # ���ת������������ɷ�
components_var = model_pca.explained_variance_  # ��ø����ɷֵķ���
components_var_ratio = model_pca.explained_variance_ratio_  # ��ø����ɷֵķ���ռ��
print(components[:2])  # ��ӡ���ǰ2�����ɷ�
print(components_var[:2])  # ��ӡ���ǰ2�����ɷֵķ���
print(components_var_ratio)  # ��ӡ����������ɷֵķ���ռ��




