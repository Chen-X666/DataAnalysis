# _*_ coding: utf-8 _*_
"""
Time:     2021/6/27 1:11
Author:   ChenXin
Version:  V 0.1
File:     Clustering.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
from sklearn import cluster
from matplotlib import pyplot as plt
import seaborn as sns

def kmeanClustering(data1,columns,x,y,num):
    data = data1[columns]
    data_reshape = data.values.reshape((data.shape[0], 1))  # 转换数据形状
    model_kmeans = cluster.KMeans(n_clusters=num, random_state=0)  # 创建KMeans模型并指定要聚类数量
    keames_result = model_kmeans.fit_predict(data_reshape)  # 建模聚类
    print(keames_result)
    print(data1)
    data1['amount'] = keames_result  # 新离散化的数据合并到原数据框
    print(data1.head(5))
    sns.stripplot(x=x,y=y, data=data1,hue='amount',jitter=True)
    #sns.set(style='whitegrid')
    plt.show()
    return keames_result
    #r.to_excel(output_file)
