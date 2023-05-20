#!/usr/bin/env python
#coding:gbk,
#  �������
 
import numpy as np  # ����numpy��
import matplotlib.pyplot as plt  # ����matplotlib��
from sklearn.cluster import KMeans  # ����sklearn����ģ��
from sklearn import metrics  # ����sklearnЧ������ģ��

# ����׼��
raw_data = np.loadtxt('cluster.txt')  # ���������ļ�
X = raw_data[:, :-1]  # �ָ�Ҫ���������
y_true = raw_data[:, -1]

# ѵ������ģ��
n_clusters = 3  # ���þ�������
model_kmeans = KMeans(n_clusters=n_clusters, random_state=0)  # ��������ģ�Ͷ���
model_kmeans.fit(X)  # ѵ������ģ��
y_pre = model_kmeans.predict(X)  # Ԥ�����ģ��
print(y_pre)
# ģ��Ч��ָ������
n_samples, n_features = X.shape  # ��������,��������
inertias = model_kmeans.inertia_  # ����������ľ������ĵľ����ܺ�
adjusted_rand_s = metrics.adjusted_rand_score(y_true, y_pre)  # �����������ָ��
mutual_info_s = metrics.mutual_info_score(y_true, y_pre)  # ����Ϣ
adjusted_mutual_info_s = metrics.adjusted_mutual_info_score(y_true, y_pre)  # ������Ļ���Ϣ
homogeneity_s = metrics.homogeneity_score(y_true, y_pre)  # ͬ�ʻ��÷�
completeness_s = metrics.completeness_score(y_true, y_pre)  # �����Ե÷�
v_measure_s = metrics.v_measure_score(y_true, y_pre)  # V-measure�÷�
silhouette_s = metrics.silhouette_score(X, y_pre, metric='euclidean')  # ƽ������ϵ��
calinski_harabaz_s = metrics.calinski_harabasz_score(X, y_pre)  # Calinski��Harabaz�÷�
print('samples: %d \t features: %d' % (n_samples, n_features))  # ��ӡ�������������������

print(70 * '-')  # ��ӡ�ָ���
print('ine\tARI\tMI\tAMI\thomo\tcomp\tv_m\tsilh\tc&h')  # ��ӡ���ָ�����
print('%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d' % (
    inertias, adjusted_rand_s, mutual_info_s, adjusted_mutual_info_s, homogeneity_s, completeness_s,
    v_measure_s,silhouette_s, calinski_harabaz_s))  # ��ӡ���ָ��ֵ
print(70 * '-')  # ��ӡ�ָ���
print('short name \t full name')  # ��ӡ�����д��ȫ������
print('ine \t inertias')
print('ARI \t adjusted_rand_s')
print('MI \t mutual_info_s')
print('AMI \t adjusted_mutual_info_s')
print('homo \t homogeneity_s')
print('comp \t completeness_s')
print('v_m \t v_measure_s')
print('silh \t silhouette_s')
print('c&h \t calinski_harabaz_s')

# ģ��Ч�����ӻ�
centers = model_kmeans.cluster_centers_  # ���������
colors = ['#4EACC5', '#FF9C34', '#4E9A06']  # ���ò�ͬ������ɫ
plt.figure()  # ��������
for i in range(n_clusters):  # ѭ�������
    index_sets = np.where(y_pre == i)  # �ҵ���ͬ�����������
    cluster = X[index_sets]  # ����ͬ������ݻ���Ϊһ�������Ӽ�
    plt.scatter(cluster[:, 0], cluster[:, 1], c=colors[i], marker='.')  # չʾ�����Ӽ��ڵ�������
    plt.plot(centers[i][0], centers[i][1], 'o', markerfacecolor=colors[i], markeredgecolor='k',
             markersize=6)  # չʾ�������Ӽ�������
plt.show()  # չʾͼ��

# ģ��Ӧ��
new_X = [1, 3.6]
cluster_label = model_kmeans.predict(np.array(new_X).reshape(1,2))
print('cluster of new data point is: %d' % cluster_label)
