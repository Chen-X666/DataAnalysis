# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
# !/usr/bin/env python
# coding:gbk,
# 决策树
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, \
    roc_curve  # 分类指标库
import prettytable
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier



def cutTree(training,text):
    X_train = training[:, 0:-1]
    y_train = training[:, -1]
    X_test = text[:, 0:-1]
    y_test = text[:, -1]
    clf = DecisionTreeClassifier(random_state=0)  # 默认是基于Gini值生成决策树
    # 拟合样本数据
    clf.fit(X_train, y_train)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    print('ccp')
    print(ccp_alphas)
    '''
    [0.00000000e+00 9.97816677e-06 9.99538787e-06 ... 2.86022505e-02
     3.79933008e-02 7.03553506e-02]
    '''
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    # return：Number of nodes in the last tree is: 1 with ccp_alpha: 0.07035535057947928

    # 绘制不同ccp_alpha取值下，clf在训练样本和测试样本上的精确度
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]
    from matplotlib import pyplot
    plt.rcParams['savefig.dpi'] = 80  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600*400
    fig, ax = pyplot.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    pyplot.savefig('ccp.jpg')
    pyplot.show()


def decideTree(training, text):
    X_train = training[:, 0:-1]
    y_train = training[:, -1]
    X_test = text[:, 0:-1]
    y_test = text[:, -1]
    # 训练分类模型
    model_tree = tree.DecisionTreeClassifier(random_state=0,ccp_alpha=0.0025)  # 建立决策树模型
    #Fit the data from training
    model_tree.fit(X_train, y_train)
    pre_y = model_tree.predict(X_test)
    pre_y2 = model_tree.predict(X_train)

    n_samples, n_features = X_train.shape
    print('samples: %d \t features: %d' % (n_samples, n_features))
    print(70 * '-')

    # 混淆矩阵
    confusion_m = confusion_matrix(y_test, pre_y)
    confusion_matrix_table = prettytable.PrettyTable()
    confusion_matrix_table.add_row(confusion_m[0, :])
    confusion_matrix_table.add_row(confusion_m[1, :])
    print('confusion matrix')
    print(confusion_matrix_table)

    # 核心评估指标
    y_score = model_tree.predict_proba(X_test)  # 获得决策树的预测概率
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
    auc_s = auc(fpr, tpr)  # AUC
    accuracy_s = accuracy_score(y_test, pre_y)  # 测试集准确率
    accuracy_s2 = accuracy_score(y_train,pre_y2) #训练集准确度
    core_metrics = prettytable.PrettyTable()  # 创建表格实例
    core_metrics.field_names = ['auc', 'accuracy_text','accuracy_train']  # 定义表格列名
    core_metrics.add_row([auc_s, accuracy_s,accuracy_s2])  # 增加数据
    print('core metrics')
    print(core_metrics)  # 打印输出核心评估指标

    # 绘制出生成的决策树
    plt.rcParams['savefig.dpi'] = 200  # 图片像素
    plt.rcParams['figure.dpi'] = 200  # 分辨率
    # 默认的像素：[6.0,4.0]，分辨率为100，图片尺寸为 600*400

    tree.plot_tree(model_tree, filled=True)

    # 子指标重要性
    feature_importance = model_tree.feature_importances_  # 获得指标重要性
    for i in range(0,len(feature_importance)):
        print('指标{}的重要性为{}'.format(i+1,feature_importance[i]))



if __name__ == "__main__":
    df = pd.DataFrame(pd.read_csv('1.csv'))
    df1 = pd.DataFrame(pd.read_csv('2.csv'))
    print(np.isnan(df).any())
    #剪枝
    #cutTree(df.values,df1.values)
    #决策树模型
    decideTree(df.values, df1.values)


