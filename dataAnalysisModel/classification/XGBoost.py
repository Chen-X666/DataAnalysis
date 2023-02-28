# _*_ coding: utf-8 _*_
"""
Time:     2022/10/15 14:31
Author:   ChenXin
Version:  V 0.1
File:     XGBoost.py
Describe:  Github link: https://github.com/Chen-X666
"""
from prettytable import prettytable
from xgboost import XGBRegressor as XGBR
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CSV, train_test_split #k折 交叉验证
from sklearn.metrics import mean_squared_error as MSE, roc_curve, auc, accuracy_score, precision_score, recall_score, \
    f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def XGboost():
    data = load_boston()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)
    model_rf =  XGBR(n_estimators = 100).fit(X_train, y_train)


    # 核心评估指标：AUC，accuracy，precision，recall，f1分数
    y_score = model_rf.predict(X_test)  # 获得决策树对每个样本点的预测概率
    print(y_score)
    print(y_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
    auc_s = auc(fpr, tpr).round(3)  # AUC
    accuracy_s = accuracy_score(y_test, y_test).round(3)  # 准确率
    precision_s = precision_score(y_test, y_test).round(3)  # 精确度
    recall_s = recall_score(y_test, y_test).round(3)  # 召回率
    f1_s = f1_score(y_test, y_test).round(3)  # F1得分
    core_metrics = prettytable.PrettyTable()  # 创建表格实例
    core_metrics.field_names = ['auc', 'accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
    core_metrics.add_row([auc_s, accuracy_s, precision_s, recall_s, f1_s])  # 增加数据
    print('{:-^60}'.format('核心评估指标'), '\n', core_metrics)
    print(fpr)
    print(tpr)
    return 0

if __name__ == '__main__':
    XGboost()