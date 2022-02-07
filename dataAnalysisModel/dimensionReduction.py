# _*_ coding: utf-8 _*_
"""
Time:     2022/2/4 17:02
Author:   ChenXin
Version:  V 0.1
File:     dimensionReduction.py
Describe:  Github link: https://github.com/Chen-X666
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
#pca降维
#主成分分析算法(PCA) Principal Component Analysis(PCA)
#线性降维算法
def pca(weight,delimension):
    pca = PCA(n_components=delimension)
    pca.fit_transform(weight)  # 载入N维
    variance = pca.explained_variance_ratio_
    var=np.cumsum(np.round(variance,3)*100)
    plt.figure(figsize=(12,6))
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Features')
    plt.title('PCA Analysis')
    plt.ylim(0,100.5)
    plt.plot(var)
    plt.show()
    return pca

#TSNE降维
def tsne(weight,delimension):
    return TSNE(n_components=delimension).fit_transform(weight)  # 载入N维