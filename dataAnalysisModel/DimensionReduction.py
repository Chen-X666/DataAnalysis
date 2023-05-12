# _*_ coding: utf-8 _*_
"""
Time:     2022/2/4 17:02
Author:   ChenXin
Version:  V 0.1
File:     DimensionReduction.py
Describe:  Github link: https://github.com/Chen-X666
"""
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
#pca降维
#主成分分析算法(PCA) Principal Component Analysis(PCA)
#线性降维算法
def PCADimensionalDeduction(weight, delimension):
    pca = PCA(n_components=delimension)
    pca.fit_transform(weight)  # 载入N维
    # variance = pca.explained_variance_ratio_
    # var=np.cumsum(np.round(variance,3)*100)
    # plt.figure(figsize=(12,6))
    # plt.ylabel('% Variance Explained')
    # plt.xlabel('# of Features')
    # plt.title('PCA Analysis')
    # plt.ylim(0,100.5)
    # plt.plot(var)
    # plt.show()
    return pca

#TSNE降维
def TSNEDimensionalDeduction(weight, delimension):
    return TSNE(n_components=delimension).fit_transform(weight)  # 载入N维

def TSNE_PCADimensionalDeduction(weight,delimension):
    newData = PCA(n_components=delimension+2).fit_transform(weight)  # 载入N维
    newData = TSNE(delimension).fit_transform(newData)
    return newData
    #plot_cluster(result, newData, numClass)


def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'ob', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^',
             'g^'] * 3
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            # print ind1
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i])

    # 绘制初始中心点
    # x1 = []
    # y1 = []
    # for ind1 in clf.cluster_centers_:
    #     try:
    #         y1.append(ind1[1])
    #         x1.append(ind1[0])
    #     except:
    #         pass
    # plt.plot(x1, y1, "rv")  # 绘制中心
    plt.show()
