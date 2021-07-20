# _*_ coding: utf-8 _*_
"""
Time:     2021/7/20 14:31
Author:   ChenXin
Version:  V 0.1
File:     SVM.py
Describe:  Github link: https://github.com/Chen-X666
"""
import numpy as np # numpy库
import pandas as pd # pandas库
import prettytable # 图表打印工具
from sklearn.svm import SVC  # SVC
from sklearn.model_selection import train_test_split # 数据切割
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.metrics import accuracy_score,auc,confusion_matrix,f1_score,precision_score,recall_score,roc_curve # 分类指标库
def SVM(X,y):
    # SVM模型实验
    # ***********************************SVM模型实验**********************************************
    print('=' * 20 + "SVM模型实现" + '=' * 20)
    # 将数据集按照4:1的比例分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    # 查看总样本量、总特征数
    n_samples, n_features = X_train.shape
    print('{:-^60}'.format('样本数据审查'))
    print('样本量: {0} | 特征数: {1}'.format(n_samples, n_features))

    # 网格检索最优参数并得到最优模型性能
    parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]  # 构建参数网格
    model_svm = SVC(probability=True)  # 选取SVC作为简单SVM模型
    model_gs = GridSearchCV(estimator=model_svm, param_grid=parameters, cv=5, n_jobs=-1,
                            scoring='precision')  # 建立交叉检验模型对象，并行数与CPU一致
    print("SVM训练耗时较久，请稍等...")
    model_gs.fit(X_train, y_train)  # 训练交叉检验模型
    print('{:-^60}'.format('模型最优化参数'))
    print('最优得分:', model_gs.best_score_)  # 获得交叉检验模型得出的最优得分,标准为precision
    print('最优参数:', model_gs.best_params_)  # 获得交叉检验模型得出的最优参数
    model_rf = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象
    model_rf.fit(X_train, y_train)  # 训练集拟合模型
    pre_y = model_rf.predict(X_test)  # 得到测试集的预测结果集合，为构建混淆矩阵做准备

    # 混淆矩阵
    TN, FP, FN, TP = confusion_matrix(y_test, pre_y).ravel()  # 获得混淆矩阵并用ravel将四个值拆开赋予TN,FP,FN,TP
    confusion_matrix_table = prettytable.PrettyTable(['', '预测2G', '预测3G'])
    confusion_matrix_table.add_row(['真实2G', TP, FN])
    confusion_matrix_table.add_row(['真实3G', FP, TN])
    print('{:-^60}'.format('混淆矩阵'), '\n', confusion_matrix_table)  # 打印

    # 核心评估指标：AUC、accuracy，precision，recall，f1分数
    y_score = model_rf.predict_proba(X_test)  # 获得决策树对每个样本点的预测概率
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
    auc_s = auc(fpr, tpr).round(3)  # AUC
    accuracy_s = accuracy_score(y_test, pre_y).round(3)  # 准确率
    precision_s = precision_score(y_test, pre_y).round(3)  # 精确度
    recall_s = recall_score(y_test, pre_y).round(3)  # 召回率
    f1_s = f1_score(y_test, pre_y).round(3)  # F1得分
    core_metrics = prettytable.PrettyTable()  # 创建表格实例
    core_metrics.field_names = ['auc', 'accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
    core_metrics.add_row([auc_s, accuracy_s, precision_s, recall_s, f1_s])  # 增加数据
    print('{:-^60}'.format('核心评估指标'), '\n', core_metrics)


