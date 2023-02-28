# _*_ coding: utf-8 _*_
"""
Time:     2021/7/20 14:29
Author:   ChenXin
Version:  V 0.1
File:     LogisticRegression.py
Describe:  Github link: https://github.com/Chen-X666
"""
import numpy as np # numpy库
import pandas as pd # pandas库
import prettytable # 图表打印工具
from sklearn import metrics
import matplotlib.pyplot as plt  # 画图工具
from sklearn import tree, linear_model  # 树、线性模型
from sklearn.model_selection import train_test_split # 数据切割
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.metrics import accuracy_score,auc,confusion_matrix,f1_score,precision_score,recall_score,roc_curve # 分类指标库

# 逻辑回归实验
#***********************************逻辑回归实验**********************************************
from dataAnalysisModelEvaluation.learningLine import plot_learning_curve


def LogisticRegress(X,y):
    print('='*20+"逻辑回归实现"+'='*20)
    # 训练集和验证集4：1切分
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)

    #查看总样本量、总特征数
    n_samples,n_features=X_train.shape
    print('{:-^60}'.format('样本数据审查'))
    print('样本量: {0} | 特征数: {1}'.format(n_samples,n_features))

    # 训练逻辑回归模型
    parameters = {'penalty':['l2'],'C':[10]} # 可优化参数
    model_t = GridSearchCV(estimator=linear_model.LogisticRegression(max_iter=1000), param_grid=parameters,verbose=0,cv=5,n_jobs=-1,scoring='f1')  # 建立交叉检验模型对象，并行数与CPU一致
    model_t.fit(X_train, y_train)  # 训练交叉检验模型
    print('{:-^60}'.format('模型最优化参数'))
    print('最优得分:', model_t.best_score_)  # 获得交叉检验模型得出的最优得分,默认是R方
    print('最优参数:', model_t.best_params_)  # 获得交叉检验模型得出的最优参数
    model = model_t.best_estimator_  # 获得交叉检验模型得出的最优模型对象
    model.fit(X_train, y_train)  # 拟合训练集

    # 预测
    y_predict = model.predict(X_test)  #
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'], index=y_test.index)
    y_test_predict_df = pd.concat([y_test, y_predict_df], axis=1)
    y_score = model.predict_proba(X_test)  # 获得决策树对每个样本点的预测概率
    print('真实值与预测值', '-' * 30, '\n', y_test_predict_df)


    # 初步评价
    print('模型','-'*30,'\n',model)
    print('模型系数','-'*30,'\n',model.coef_)
    print('模型截距','-'*30,'\n',model.intercept_)
    print('模型得分','-'*30,'\n',model.score(X_test,y_test))


    # 评估模型
    drawROC(y_test, y_score)

    # 混淆矩阵
    drawConfusionMatrix(y_test, y_predict)

    # 核心评估指标：accuracy，precision，recall，f1分数
    drawROC(y_test,y_predict,y_score)

    # 学习曲线
    plot_learning_curve(model, X_train, X_test, y_train, y_test, score=accuracy_score)



# 模型数值评估
def valueEvaluation(y_test,y_predict):
    accuracy = metrics.accuracy_score(y_test, y_predict) # 精度
    confusionmatrix = metrics.confusion_matrix(y_test, y_predict) # 混淆矩阵
    target_names = ['class 0', 'class 1'] # 两个类别
    classifyreport = metrics.classification_report(y_test, y_predict,target_names=target_names) # 分类结果报告
    print('分类准确率 ',accuracy) # 混淆矩阵对角线元素之和/所有元素之和
    print('混淆矩阵 \n', confusionmatrix)
    print('分类结果报告 \n', classifyreport)
    # 核心评估指标：accuracy，precision，recall，f1分数
    accuracy_s = accuracy_score(y_test, y_predict).round(3)  # 准确率
    precision_s = precision_score(y_test, y_predict).round(3)  # 精确度
    recall_s = recall_score(y_test, y_predict).round(3)  # 召回率
    f1_s = f1_score(y_test, y_predict).round(3)  # F1得分
    core_metrics = prettytable.PrettyTable()  # 创建表格实例
    core_metrics.field_names = ['accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
    core_metrics.add_row([accuracy_s, precision_s, recall_s, f1_s])  # 增加数据
    print('{:-^60}'.format('核心评估指标'), '\n', core_metrics)


# 混淆矩阵
def drawConfusionMatrix(y_test, pre_y):
    TN, FP, FN, TP = confusion_matrix(y_test, pre_y).ravel()  # 获得混淆矩阵并用ravel将四个值拆开赋予TN,FP,FN,TP
    confusion_matrix_table = prettytable.PrettyTable(['', '预测0', '预测1'])
    confusion_matrix_table.add_row(['真实0', TP, FN])
    confusion_matrix_table.add_row(['真实1', FP, TN])
    print('{:-^60}'.format('混淆矩阵'), '\n', confusion_matrix_table)

def drawROC(y_test,y_score):
    # 核心评估指标：accuracy，precision，recall，f1分数
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
    # auc_s = auc(fpr, tpr).round(3)  # AUC
    # 打印ROC曲线
    plt.subplot(1, 2, 1)  # 第一个子网格
    plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')  # 画出随机状态下的准确率线
    plt.title('ROC')  # 子网格标题
    plt.xlabel('false positive rate')  # X轴标题
    plt.ylabel('true positive rate')  # y轴标题
    plt.legend(loc=0)
    plt.show()  # 展示图形




