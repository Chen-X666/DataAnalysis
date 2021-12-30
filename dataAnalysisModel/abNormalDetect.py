# _*_ coding: utf-8 _*_
"""
Time:     2021/11/1 23:40
Author:   ChenXin
Version:  V 0.1
File:     abNormalDetect.py
Describe:  Github link: https://github.com/Chen-X666
"""
# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scipy import stats

from sklearn.svm import OneClassSVM  # 导入OneClassSVM
import numpy as np  # 导入numpy库
import matplotlib.pyplot as plt  # 导入Matplotlib
from mpl_toolkits.mplot3d import Axes3D  # 导入3D样式库
import pandas as pd
from sklearn.ensemble import VotingClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
#from sklearn.preprocessing import OrdinalEncoder
#由于版本问题，无法导入OrdinalEncoder，导入LabelEncoder可以实现同样效果，但是一次只能处理一列数据
#from sklearn.preprocessing import LabelEncode

#OneClassSVM不是一种outlier detection方法，而是一种novelty detection方法：它的训练集不应该掺杂异常点，
# 因为模型可能会去匹配这些异常点。 但在数据维度很高，或者对相关数据分布没有任何假设的情况下，
# OneClassSVM也可以作为一种很好的outlier detection方法。
def OneClassSvm(raw_data):
    print(raw_data.shape)
    train_set = raw_data[:900, :]  # 训练集
    test_set = raw_data[900:, :]  # 测试集
    train_set = raw_data
    test_set = raw_data

    # 异常数据检测
    model_onecalsssvm = OneClassSVM(nu=0.01, kernel="rbf")  # 创建异常检测算法模型对象
    model_onecalsssvm.fit(train_set)  # 训练模型
    pre_test_outliers = model_onecalsssvm.predict(test_set)  # 异常检测,1标识正常数据，-1标识异常数据
    print(pre_test_outliers.shape)

    # 异常结果统计
    toal_test_data = np.hstack((test_set, pre_test_outliers.reshape(test_set.shape[0], 1)))  # 将测试集和检测结果合并
    # vstack()#在竖直方向拼接数组
    normal_test_data = toal_test_data[toal_test_data[:, -1] == 1]  # 获得异常检测结果中正常数据集
    outlier_test_data = toal_test_data[toal_test_data[:, -1] == -1]  # 获得异常检测结果中异常数据
    n_test_outliers = outlier_test_data.shape[0]  # 获得异常的结果数量
    total_count_test = toal_test_data.shape[0]  # 获得测试集样本量
    print('outliers: {0}/{1}'.format(n_test_outliers, total_count_test))  # 输出异常的结果数量
    print('{:*^60}'.format(' all result data (limit 5) '))  # 打印标题
    print(toal_test_data[:5])  # 打印输出前5条合并后的数据集

    # 异常检测结果展示
    plt.style.use('ggplot')  # 使用ggplot样式库
    fig = plt.figure()  # 创建画布对象
    ax = Axes3D(fig)  # 将画布转换为3D类型
    s1 = ax.scatter(normal_test_data[:, 2], normal_test_data[:, 3], normal_test_data[:, 4], s=100, edgecolors='k',
                    c='g',
                    marker='o')  # 画出正常样本点
    s2 = ax.scatter(outlier_test_data[:, 2], outlier_test_data[:, 3], outlier_test_data[:, 4], s=100, edgecolors='k',
                    c='r',
                    marker='o')  # 画出异常样本点
    ax.w_xaxis.set_ticklabels([])  # 隐藏x轴标签，只保留刻度线
    ax.w_yaxis.set_ticklabels([])  # 隐藏y轴标签，只保留刻度线
    ax.w_zaxis.set_ticklabels([])  # 隐藏z轴标签，只保留刻度线
    ax.legend([s1, s2], ['normal points', 'outliers'], loc=0)  # 设置两类样本点的图例
    plt.title('novelty detection')  # 设置图像标题
    plt.show()  # 展示图像

    return outlier_test_data

def isoForest():

    rng = np.random.RandomState(42)

    # 构造训练样本
    n_samples = 200  # 样本总数
    outliers_fraction = 0.25  # 异常样本比例
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)

    X = 0.3 * rng.randn(n_inliers // 2, 2)
    X_train = np.r_[X + 2, X - 2]  # 正常样本
    X_train = np.r_[X_train, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]  # 正常样本加上异常样本

    # fit the model
    clf = IsolationForest(max_samples=n_samples, random_state=rng, contamination=outliers_fraction)
    clf.fit(X_train)
    # y_pred_train = clf.predict(X_train)
    scores_pred = clf.decision_function(X_train)
    threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)  # 根据训练样本中异常样本比例，得到阈值，用于绘图

    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(-7, 7, 50), np.linspace(-7, 7, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    #plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7), cmap=plt.cm.Blues_r)  # 绘制异常点区域，值从最小的到阈值的那部分
    a = plt.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red')  # 绘制异常点区域和正常点区域的边界
    plt.contourf(xx, yy, Z, levels=[threshold, Z.max()], colors='palevioletred')  # 绘制正常点区域，值从阈值到最大的那部分

    b = plt.scatter(X_train[:-n_outliers, 0], X_train[:-n_outliers, 1], c='white',
                    s=20, edgecolor='k')
    c = plt.scatter(X_train[-n_outliers:, 0], X_train[-n_outliers:, 1], c='black',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((-7, 7))
    plt.ylim((-7, 7))
    plt.legend([a.collections[0], b, c],
               ['learned decision function', 'true inliers', 'true outliers'],
               loc="upper left")
    plt.show()

def GBDT():
    # 样本均衡，过抽样处理
    # 由于环境问题，无法安装该库，因此没有进行样本均衡，但是模型效果相差不大，主要是因为样本不均衡问题并#不严重
    # model_smote = SMOTE()
    # x_smote_resampled,y_smote_resampled = model_smote.fit_sample(X_train,y_train)
    # 模型训练，交叉检验
    model_rf = RandomForestClassifier(max_features=0.8, random_state=0)
    model_gdbc = GradientBoostingClassifier(max_features=0.8, random_state=0)
    estimators = [('randomforest', model_rf), ('gradientboosting', model_gdbc)]  # 建立组合评估器列表
    model_vot = VotingClassifier(estimators=estimators, voting="soft", weights=[0.9, 1.2], n_jobs=-1)
    cv = StratifiedKFold(5, random_state=2)
    cv_score = cross_val_score(model_gdbc, X_train, y_train, cv=cv)
    # cv_score ---array([0.7318018 , 0.79157476, 0.84306236, 0.80888841, 0.74468658])
    model_vot.score(X_train, y_train)


if __name__ == '__main__':
    # 数据准备
    raw_data = np.loadtxt('../数据分析课程代码库/data/outlier.txt', delimiter=' ')  # 读取数据
    OneClassSvm(raw_data)

