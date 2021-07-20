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
def LogisticRegress(X,y):
    print('='*20+"逻辑回归实现"+'='*20)
    # 训练集和验证集4：1切分
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)

    #查看总样本量、总特征数
    n_samples,n_features=X_train.shape
    print('{:-^60}'.format('样本数据审查'))
    print('样本量: {0} | 特征数: {1}'.format(n_samples,n_features))

    # 训练逻辑回归模型并初步评价
    logisticregression = linear_model.LogisticRegression(max_iter=1000,penalty='l2',C=10)
    model =logisticregression.fit(X_train,y_train)
    print('模型','-'*30,'\n',model)
    print('模型系数','-'*30,'\n',model.coef_)
    print('模型截距','-'*30,'\n',model.intercept_)
    print('模型得分','-'*30,'\n',model.score(X_test,y_test))

    # 打印真实值与预测值以查看并初步评判其精度
    y_predict = model.predict(X_test) # 预测
    y_predict_df = pd.DataFrame(y_predict,columns=['y_predict'],index=y_test.index)
    y_test_predict_df = pd.concat([y_test,y_predict_df],axis = 1)
    print('真实值与预测值','-'*30,'\n',y_test_predict_df)

    # 评估模型
    accuracy = metrics.accuracy_score(y_test, y_predict) # 精度
    confusionmatrix = metrics.confusion_matrix(y_test, y_predict) # 混淆矩阵
    target_names = ['class 0', 'class 1'] # 两个类别
    classifyreport = metrics.classification_report(y_test, y_predict,target_names=target_names) # 分类结果报告
    print('分类准确率 ',accuracy) # 混淆矩阵对角线元素之和/所有元素之和
    print('混淆矩阵 \n', confusionmatrix)
    print('分类结果报告 \n', classifyreport)

    # 优化模型,选择模型最佳参数
    parameters = {'penalty':['l2'],'C':[0.01, 0.1, 1, 10]} # 可优化参数
    model_gs = GridSearchCV(estimator=linear_model.LogisticRegression(max_iter=1000), param_grid=parameters,verbose=0,cv=5,n_jobs=-1,scoring='accuracy')  # 建立交叉检验模型对象，并行数与CPU一致
    model_gs.fit(X_train, y_train)  # 训练交叉检验模型
    print('{:-^60}'.format('模型最优化参数'))
    print('最优得分:', model_gs.best_score_)  # 获得交叉检验模型得出的最优得分,评判标准为accuracy
    print('最优参数:', model_gs.best_params_)  # 获得交叉检验模型得出的最优参数
    model_lg = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象
    model_lg.fit(X_train,y_train)  # 拟合训练集
    pre_y = model_lg.predict(X_test) # 得到测试集的预测结果集合，为构建混淆矩阵做准备

    # 混淆矩阵
    TN,FP,FN,TP=confusion_matrix(y_test,pre_y).ravel() # 获得混淆矩阵并用ravel将四个值拆开赋予TN,FP,FN,TP
    confusion_matrix_table=prettytable.PrettyTable(['','预测2G','预测3G'])
    confusion_matrix_table.add_row(['真实2G',TP,FN])
    confusion_matrix_table.add_row(['真实3G',FP,TN])
    print('{:-^60}'.format('混淆矩阵'),'\n',confusion_matrix_table)

    # 核心评估指标：accuracy，precision，recall，f1分数
    y_score = model_lg.predict_proba(X_test)  # 获得决策树对每个样本点的预测概率
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
    # auc_s = auc(fpr, tpr).round(3)  # AUC
    accuracy_s = accuracy_score(y_test, pre_y).round(3)  # 准确率
    precision_s = precision_score(y_test, pre_y).round(3)  # 精确度
    recall_s = recall_score(y_test, pre_y).round(3) # 召回率
    f1_s = f1_score(y_test, pre_y).round(3)  # F1得分
    core_metrics = prettytable.PrettyTable()  # 创建表格实例
    core_metrics.field_names = ['accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
    core_metrics.add_row([accuracy_s, precision_s, recall_s, f1_s])  # 增加数据
    print('{:-^60}'.format('核心评估指标'),'\n',core_metrics)
    # 打印ROC曲线
    plt.subplot(1, 2, 1)  # 第一个子网格
    plt.plot(fpr, tpr, label='ROC')  # 画出ROC曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')  # 画出随机状态下的准确率线
    plt.title('ROC')  # 子网格标题
    plt.xlabel('false positive rate')  # X轴标题
    plt.ylabel('true positive rate')  # y轴标题
    plt.legend(loc=0)
    plt.show()  # 展示图形

