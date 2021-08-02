# _*_ coding: utf-8 _*_
"""
Time:     2021/7/20 14:25
Author:   ChenXin
Version:  V 0.1
File:     KNN.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 画图工具
import prettytable # 图表打印工具
from sklearn import metrics
from sklearn.model_selection import train_test_split # 数据切割
from sklearn.neighbors import KNeighborsClassifier   # K近邻
from sklearn.model_selection import cross_val_score  # 交叉检验
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn import tree, linear_model  # 树、线性模型
from sklearn.metrics import accuracy_score,auc,confusion_matrix,f1_score,precision_score,recall_score,roc_curve # 分类指标库

def KNN(X,y):
    # KNN模型实验
    # ***********************************KNN模型实验**********************************************
    # ******************************KNN的第一种模式，找最优K*****************************************
    print('=' * 20 + "KNN实验" + '=' * 20)
    # 将数据集按照4:1的比例分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    # 查看总样本量、总特征数
    n_samples, n_features = X_train.shape
    print('{:-^60}'.format('样本数据审查'))
    print('样本量: {0} | 特征数: {1}'.format(n_samples, n_features))

    # 构建一个最简KNN模型
    knn_clf = KNeighborsClassifier()

    # 通过对构建不同超参K的KNN模型通过交叉检验分别求精度值，得到精度最高的k值
    k_range = range(1, 50)  # K为1-49
    k_score = []
    max = 0  # 精度最高时的分数
    index = 0  # # 精度最高时的K值
    for i in k_range:
        knn = KNeighborsClassifier(n_neighbors=i)
        score = cross_val_score(knn, X, y, cv=10, scoring='accuracy')  # 10-fold交叉检验得到分数K从1到50的分数（利用accuracy评估）
        if score.mean() > max:
            max = score.mean()
            index = i
        k_score.append(score.mean())

    print('K为{}时，精度取得最高：{}'.format(index, max))  # 打印结果
    plt.plot(k_range, k_score)  # 画图
    plt.xlabel('Value of K for KNN')
    plt.ylabel('accuracy')
    plt.show()

    # 模型评价
    ## 用平均方差(Mean squared error)判断模型好坏
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        loss = -cross_val_score(knn, X, y, cv=10, scoring='neg_mean_squared_error')
        k_scores.append(loss.mean())

    plt.plot(k_range, k_scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated MSE')
    plt.show()

    # ******************KNN的第一种模式，找最优{'n_neighbors''p''weights'}*******************************
    # 网格检索最优参数并得到最优模型性能
    param_search = [  # 网格参数
        {"weights": ["uniform"], "n_neighbors": [i for i in range(1, 11)]},
        {"weights": ["distance"], "n_neighbors": [i for i in range(1, 11)], "p": [i for i in range(1, 6)]}
    ]

    # 定义网格搜索的对象grid_search，
    model_gs = GridSearchCV(estimator=knn_clf, param_grid=param_search, cv=5, n_jobs=-1)  # 建立交叉检验模型对象，并行数与CPU一致
    model_gs.fit(X_train, y_train)  # 训练交叉检验模型
    print('{:-^60}'.format('模型最优化参数'))
    print('最优得分:', model_gs.best_score_)  # 获得交叉检验模型得出的最优得分,默认是R方
    print('最优参数:', model_gs.best_params_)  # 获得交叉检验模型得出的最优参数
    model_knn = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象

    #为什么还要继续遍历？
    test_accuracy = []
    # n_neighbors取值从1到50
    neighbors_settings = range(1, 50)
    for n_neighbors in neighbors_settings:
        # 构建模型
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', p=1)
        # 使用K折交叉验证模块(把样本分成5份，每一份都为训练集，得到精确度再求平均值）
        scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
        # 将5次的预测准确率打印出
        print(scores)
        # 将5次的预测准确平均率打印出
        print(scores.mean())
        # 记录测试集精度
        test_accuracy.append(scores.mean())

    # 画图
    plt.figure()
    plt.title("KNN_score Graph")
    plt.plot(neighbors_settings, test_accuracy, color="g", label="test accuracy")  # test accuracy类别标签
    plt.ylabel("Accuracy")  # y坐标
    plt.xlabel("n_neighbors")  # x坐标
    plt.grid()
    plt.legend(loc='best')
    plt.show()

    # 得到最优参为2构建模型
    knn = KNeighborsClassifier(n_neighbors=2, weights='distance', p=1)
    # 使用K折交叉验证模块(把样本分成5份，每一份都为训练集，得到精确度再求平均值）
    scores = cross_val_score(knn, X, y, cv=5, scoring='accuracy')
    # 将5次的预测准确率打印出
    print(scores)
    # 将5次的预测准确平均率打印出
    print(scores.mean())


