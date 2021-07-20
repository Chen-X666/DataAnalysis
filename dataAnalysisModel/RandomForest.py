# _*_ coding: utf-8 _*_
"""
Time:     2021/7/20 13:24
Author:   ChenXin
Version:  V 0.1
File:     RandomForest.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # 画图工具
from sklearn.model_selection import train_test_split # 数据切割
import prettytable # 图表打印工具
from sklearn.model_selection import GridSearchCV  # 网格搜索
from subprocess import call
from sklearn.metrics import accuracy_score,auc,confusion_matrix,f1_score,precision_score,recall_score,roc_curve # 分类指标库
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn import tree, linear_model  # 树、线性模型
def decisionTree(X,y):
    # 随机森林实验
    # ***********************************决策树实验**********************************************
    print('=' * 20 + "随机森林实验" + '=' * 20)
    # 将数据集按照4:1的比例分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=0)

    # 查看总样本量、总特征数
    n_samples, n_features = X_train.shape
    print('{:-^60}'.format('样本数据审查'))
    print('样本量: {0} | 特征数: {1}'.format(n_samples, n_features))

    # 网格检索获得最优参及最优模型
    model_RF = RandomForestClassifier(random_state=0)  # 建立随机森林分类模型对象
    parameters = {'n_estimators': range(10, 60, 5),  # 要优化的参数信息
                  'max_depth': range(10, 20, 1),
                  'max_features': range(4, 10, 2)
                  }
    model_gs = GridSearchCV(estimator=model_RF, param_grid=parameters, cv=5, n_jobs=-1)  # 建立交叉检验模型对象，并行数与CPU一致
    model_gs.fit(X_train, y_train)  # 训练交叉检验模型
    print('{:-^60}'.format('模型最优化参数'))
    print('最优得分:', model_gs.best_score_)  # 获得交叉检验模型得出的最优得分,默认是R方
    print('最优参数:', model_gs.best_params_)  # 获得交叉检验模型得出的最优参数
    model_rf = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象
    model_rf.fit(X_train, y_train)  # 拟合训练集
    pre_y = model_rf.predict(X_test)  # 得到测试集的预测结果集合，为构建混淆矩阵做准备

    # 混淆矩阵
    TN, FP, FN, TP = confusion_matrix(y_test, pre_y).ravel()  # 获得混淆矩阵并用ravel将四个值拆开赋予TN,FP,FN,TP
    confusion_matrix_table = prettytable.PrettyTable(['', '预测2G', '预测3G'])
    confusion_matrix_table.add_row(['真实2G', TP, FN])
    confusion_matrix_table.add_row(['真实3G', FP, TN])
    print('{:-^60}'.format('混淆矩阵'), '\n', confusion_matrix_table)

    # 核心评估指标：AUC，accuracy，precision，recall，f1分数
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

    # ROC曲线
    plt.subplot(1, 2, 1)  # 第一个子网格
    plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')  # 画出随机状态下的准确率线
    plt.title('ROC')  # 子网格标题
    plt.xlabel('false positive rate')  # X轴标题
    plt.ylabel('true positive rate')  # y轴标题
    plt.legend(loc=0)

    # 特征重要性
    names_list = X.columns  # 分类模型维度列表
    feature_importance = model_rf.feature_importances_  # 获得特征重要性
    plt.subplot(1, 2, 2)  # 第二个子网格
    plt.bar(np.arange(feature_importance.shape[0]), feature_importance)  # 画出条形图
    plt.title('feature importance')  # 子网格标题
    plt.xlabel('features')  # x轴标题
    plt.ylabel('importance')  # y轴标题
    plt.xticks(range(len(names_list)), names_list, rotation=90, size=8)
    plt.suptitle('classification result')  # 图形总标题
    plt.show()  # 展示图形

    # 得到树图
    estimator = model_rf.estimators_[1]
    tree.export_graphviz(estimator, feature_names=names_list, out_file='tree.dot',
                         filled=True, rounded=True, special_characters=True, class_names=["0", "1"], max_depth=4)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
