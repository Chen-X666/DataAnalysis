# _*_ coding: utf-8 _*_
"""
Time:     2021/7/20 13:24
Author:   ChenXin
Version:  V 0.1
File:     RandomForest.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
import pydotplus
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt  # 画图工具
from sklearn.model_selection import train_test_split, learning_curve  # 数据切割
import prettytable # 图表打印工具
from sklearn.model_selection import GridSearchCV  # 网格搜索
from subprocess import call
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, \
    mean_squared_error  # 分类指标库
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn import tree, linear_model  # 树、线性模型

from dataAnalysisModel.classification.classificationMetrics import valueEvaluation, ROC, ConfusionMatrix
from dataAnalysisModelEvaluation import drawLine
from dataAnalysisModelEvaluation.learningLine import plot_learning_curve


def decisionTree(X_train, X_test, y_train, y_test):
    print('{:-^60}'.format('随机森林实验'))
    # 查看总样本量、总特征数
    n_samples, n_features = X_train.shape
    print('{:-^60}'.format('样本数据审查'))
    print('样本量: {0} | 特征数: {1}'.format(n_samples, n_features))

    # 网格检索获得最优参及最优模型
    model_RF = RandomForestClassifier(random_state=0,class_weight='balanced',criterion='gini')  # 建立随机森林分类模型对象
    parameters = {'n_estimators': range(10, 100, 10),  # 要优化的参数信息
                  'max_depth': range(2, 18, 2),
                 #'min_samples_split': range(10, 500, 20),
                  'max_features': range(2, 6, 1)
                  }
    parameters = {'n_estimators': [20],
                  'max_depth': [6],
                  # 'min_samples_split': range(10, 500, 20),
                  'max_features': [5]
                  }
    model_gs = GridSearchCV(estimator=model_RF, param_grid=parameters, cv=10, n_jobs=-1,scoring='roc_auc')  # 建立交叉检验模型对象，并行数与CPU一致
    model_gs.fit(X_train, y_train)  # 训练交叉检验模型
    print('{:-^60}'.format('模型最优化参数'))
    print('最优得分:', model_gs.best_score_)  # 获得交叉检验模型得出的最优得分,默认是R方
    print('最优参数:', model_gs.best_params_)  # 获得交叉检验模型得出的最优参数
    model_rf = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象
    model_rf.fit(X_train, y_train)  # 拟合训练集
    pre_y = model_rf.predict(X_test)  # 得到测试集的预测结果集合，为构建混淆矩阵做准备
    y_score = model_rf.predict_proba(X_test)  # 获得决策树对每个样本点的预测概率
    print(X_train.columns)
    ROC(modelName='Random Forest',y_test=y_test, y_score=y_score)

    featureImportant(X_train, model_rf)

    # 混淆矩阵
    ConfusionMatrix(y_test, pre_y)

    # 核心评估指标：accuracy，precision，recall，f1分数
    # valueEvaluation(y_test, pre_y, y_score)

    #绘制学习曲线，调到最优参后再绘制
    #plot_learning_curve(model_gs, X_train, X_test, y_train, y_test)
    #绘制曲线

    #drawLine.drawTestLiner(model_gs, X_train, X_test, y_train, y_test,line=precision_score)
    # 得到树图
    # estimator = model_rf.esptimators_[1]
    # tree.export_graphviz(estimator, feature_names=X_train.columns, out_file='tree.dot',
    #                      filled=True, rounded=True, special_characters=True, class_names=["0", "1"], max_depth=4)
    #
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    # fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    # cn = ['setosa', 'versicolor', 'virginica']
    # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
    # tree.plot_tree(model_rf,
    #                feature_names=fn,
    #                class_names=cn,
    #                filled=True)
    # fig.savefig('imagename.png')
    import joblib
    joblib.dump(model_rf, "my_random_forest.joblib")  # save

    return model_rf



def featureImportant(X,model):
    # 特征重要性
    # names_list = X.columns  # 分类模型维度列表
    # feature_importance = model_rf.feature_importances_  # 获得特征重要性
    # plt.subplot(1, 2, 2)  # 第二个子网格
    # plt.bar(np.arange(feature_importance.shape[0]), feature_importance)  # 画出条形图
    # plt.title('feature importance')  # 子网格标题
    # plt.xlabel('features')  # x轴标题
    # plt.ylabel('importance')  # y轴标题
    # plt.xticks(range(len(names_list)), names_list, rotation=90, size=8)
    # plt.suptitle('classification result')  # 图形总标题
    # plt.show()  # 展示图形

    coef_lr = pd.DataFrame({'var': X.columns,
                            'coef': model.feature_importances_.flatten()
                            })

    index_sort = np.abs(coef_lr['coef']).sort_values().index
    coef_lr_sort = coef_lr.loc[index_sort, :]
    plt.bar(coef_lr_sort['var'], coef_lr_sort['coef'])  # 画出条形图
    plt.title('feature importance')  # 网格标题
    plt.xlabel('features')  # x轴标题
    plt.ylabel('importance')  # y轴标题
    print((coef_lr_sort['coef']).to_list())
    plt.xticks(range(len((coef_lr_sort['var']).to_list())), (coef_lr_sort['var']).to_list(), rotation=10, size=8)
    # 显示数字
    for a, b in zip(coef_lr_sort['var'], coef_lr_sort['coef']):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=8)
    plt.show()  # 展示图形
    print(coef_lr_sort)


def confusionMatrixTable(y_test,pre_y):
    # 混淆矩阵
    TN, FP, FN, TP = confusion_matrix(y_test, pre_y).ravel()  # 获得混淆矩阵并用ravel将四个值拆开赋予TN,FP,FN,TP
    confusion_matrix_table = prettytable.PrettyTable(['', '预测0', '预测1'])
    confusion_matrix_table.add_row(['真实0', TP, FN])
    confusion_matrix_table.add_row(['真实1', FP, TN])
    print('{:-^60}'.format('混淆矩阵'), '\n', confusion_matrix_table)

def coreMetrics():
    print()

