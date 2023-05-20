#coding:gbk,
#�쳣���
#OneClassSVM����һ��outlier detection����������һ��novelty detection����������ѵ������Ӧ�ò����쳣�㣬��Ϊģ�Ϳ��ܻ�ȥƥ����Щ�쳣�㡣 ��������ά�Ⱥܸߣ����߶�������ݷֲ�û���κμ��������£�OneClassSVMҲ������Ϊһ�ֺܺõ�outlier detection������

from sklearn.svm import OneClassSVM  # ����OneClassSVM
import numpy as np  # ����numpy��
import matplotlib.pyplot as plt  # ����Matplotlib
from mpl_toolkits.mplot3d import Axes3D  # ����3D��ʽ��
import pandas as pd


# ����׼��
raw_data = np.loadtxt('data/outlier.txt', delimiter=' ')  # ��ȡ����


print(raw_data.shape)
train_set = raw_data[:900, :]  # ѵ����
test_set = raw_data[900:, :]  # ���Լ�

# �쳣���ݼ��
model_onecalsssvm = OneClassSVM(nu=0.3, kernel="rbf")  # �����쳣����㷨ģ�Ͷ���
model_onecalsssvm.fit(train_set)  # ѵ��ģ��
pre_test_outliers = model_onecalsssvm.predict(test_set)  # �쳣���,1��ʶ�������ݣ�-1��ʶ�쳣����
print(pre_test_outliers.shape)

# �쳣���ͳ��
toal_test_data = np.hstack((test_set, pre_test_outliers.reshape(test_set.shape[0], 1)))  # �����Լ��ͼ�����ϲ�
#vstack()#����ֱ����ƴ������
normal_test_data = toal_test_data[toal_test_data[:, -1] == 1]  # ����쳣��������������ݼ�
outlier_test_data = toal_test_data[toal_test_data[:, -1] == -1]  # ����쳣��������쳣����
n_test_outliers = outlier_test_data.shape[0]  # ����쳣�Ľ������
total_count_test = toal_test_data.shape[0]  # ��ò��Լ�������
print('outliers: {0}/{1}'.format(n_test_outliers, total_count_test))  # ����쳣�Ľ������
print('{:*^60}'.format(' all result data (limit 5) '))  # ��ӡ����
print(toal_test_data[:5])  # ��ӡ���ǰ5���ϲ�������ݼ�

# �쳣�����չʾ
plt.style.use('ggplot') # ʹ��ggplot��ʽ��
fig = plt.figure()  # ������������
ax = Axes3D(fig)  # ������ת��Ϊ3D����
s1 = ax.scatter(normal_test_data[:, 0], normal_test_data[:, 1], normal_test_data[:, 2], s=100, edgecolors='k', c='g',
                marker='o')  # ��������������
s2 = ax.scatter(outlier_test_data[:, 0], outlier_test_data[:, 1], outlier_test_data[:, 2], s=100, edgecolors='k', c='r',
                marker='o')  # �����쳣������
ax.w_xaxis.set_ticklabels([])  # ����x���ǩ��ֻ�����̶���
ax.w_yaxis.set_ticklabels([])  # ����y���ǩ��ֻ�����̶���
ax.w_zaxis.set_ticklabels([])  # ����z���ǩ��ֻ�����̶���
ax.legend([s1, s2], ['normal points', 'outliers'], loc=0)  # ���������������ͼ��
plt.title('novelty detection')  # ����ͼ�����
plt.show()  # չʾͼ��
