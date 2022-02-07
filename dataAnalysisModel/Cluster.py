# _*_ coding: utf-8 _*_
"""
Time:     2022/2/6 15:06
Author:   ChenXin
Version:  V 0.1
File:     Cluster.py
Describe:  Github link: https://github.com/Chen-X666
"""
#kmean
from sklearn.cluster import KMeans, DBSCAN


def KMeanCluster(k,data):
    # 这里也可以选择随机初始化init="random"
    clf = KMeans(n_clusters=k, max_iter=10000, init="k-means++", tol=1e-6).fit(data)
    return clf

def DBSCANCluster(eps,min_samples,data):
    clf = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    return clf
