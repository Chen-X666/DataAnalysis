# _*_ coding: utf-8 _*_
"""
Time:     2021/7/20 14:29
Author:   ChenXin
Version:  V 0.1
File:     LogisticRegression.py
Describe:  Github link: https://github.com/Chen-X666
"""
import seaborn as sns
import numpy as np # numpy库
import pandas as pd # pandas库
import prettytable # 图表打印工具
from sklearn import metrics
import matplotlib.pyplot as plt  # 画图工具
from sklearn import tree, linear_model  # 树、线性模型
from sklearn.model_selection import train_test_split # 数据切割
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, \
    roc_auc_score  # 分类指标库

from dataAnalysisModel.classification.classificationMetrics import ConfusionMatrix, ROC, valueEvaluation
# 逻辑回归实验
#***********************************逻辑回归实验**********************************************
from dataAnalysisModelEvaluation.learningLine import plot_learning_curve



def LogisticRegress(X_train, X_test, y_train, y_test):
    print('='*20+"逻辑回归实现"+'='*20)
    # 训练集和验证集4：1切分
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)

    #查看总样本量、总特征数
    n_samples,n_features=X_train.shape
    print('{:-^60}'.format('样本数据审查'))
    print('样本量: {0} | 特征数: {1}'.format(n_samples,n_features))

    # 训练逻辑回归模型
    parameters = {'penalty':['l1','l2'],'C':[0.001,0.01,0.1,1,10,100],'solver':['lbfgs']} # 可优化参数
    parameters = {'penalty': ['l2'], 'C': [0.1], 'solver': ['lbfgs']}  # 最优参数
    model_t = GridSearchCV(estimator=linear_model.LogisticRegression(max_iter=1000), param_grid=parameters,verbose=0,cv=10,n_jobs=-1,scoring='roc_auc')  # 建立交叉检验模型对象，并行数与CPU一致
    model_t.fit(X_train, y_train)  # 训练交叉检验模型
    print('{:-^60}'.format('模型最优化参数'))
    print('最优得分:', model_t.best_score_)  # 获得交叉检验模型得出的最优得分,默认是R方
    print('最优参数:', model_t.best_params_)  # 获得交叉检验模型得出的最优参数
    model = model_t.best_estimator_  # 获得交叉检验模型得出的最优模型对象
    model.fit(X_train, y_train)  # 拟合最优参数的模型

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

    print(y_test)
    print(y_score)
    # # 评估模型
    ROC(modelName='Logistic Regression ROC',y_test=y_test, y_score=y_score)
    #
    # # 特征重要性
    # featureImportant(X_train,model)
    #
    # # 混淆矩阵
    ConfusionMatrix(y_test, y_predict)
    #
    # # 核心评估指标：accuracy，precision，recall，f1分数
    valueEvaluation(y_test,y_predict,y_score)
    #
    # # 学习曲线
    # plot_learning_curve(model, X_train, X_test, y_train, y_test)

    return model

def featureImportant(X,model):
    sns.set()
    coef_lr = pd.DataFrame({'var': X.columns,
                            'coef': model.coef_.flatten()
                            })

    index_sort = np.abs(coef_lr['coef']).sort_values().index
    coef_lr_sort = coef_lr.loc[index_sort, :]
    plt.bar(coef_lr_sort['var'], coef_lr_sort['coef'])  # 画出条形图
    plt.title('feature importance')  # 网格标题
    plt.xlabel('features')  # x轴标题
    plt.ylabel('importance')  # y轴标题
    print((coef_lr_sort['coef']).to_list())
    plt.xticks(range(len((coef_lr_sort['var']).to_list())),(coef_lr_sort['var']).to_list(), rotation=10, size=8)
    #显示数字
    for a, b in zip(coef_lr_sort['var'], coef_lr_sort['coef']):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=8)
    plt.show()  # 展示图形
    print(coef_lr_sort)


# Penalty
# 'l1' penalty refers to LASSO regression (more strict), and 'l2' to Ridge regression (less strict, but easiear to solve). My advice: Start with LASSO, if it doesn't work or you are not happy with the results, move to Ridge. Large problems (over 10 million points) might require the use of Ridge regression in home computers.
# Penalty constant
# This refers to how much to weight the error in prediction versus the regularization (penalty) error. When optimizing the parameters, a penalization constant will try to optimise the following:
# 𝐸𝑟𝑟𝑜𝑟=𝐸𝑟𝑟𝑜𝑟𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛+𝐶×𝐸𝑟𝑟𝑜𝑟𝑝𝑒𝑛𝑎𝑙𝑡𝑦
# So the
# 𝐶
# constant will balance both objectives. This is a tunable parameter, meaning we need to search for what leaves us happy. In general, if you think the model is being too strict, then reduce C; if it is being too lax, increase it.
#
# Class weighting
# Most interesting problems are unbalanced. This means the interesting class (Default in our case) has less cases than the opposite class. Models optimise the sum over all cases, so if we minimize the error, which class do you think will be better classified?
# This means we need to balance the classes to make them equal. Luckily for us, Scikit-Learn includes automatic weighting that assigns the same error to both classes. The error becomes the following:
# 𝐸𝑟𝑟𝑜𝑟=𝑊𝑒𝑖𝑔ℎ𝑡1×𝐸𝑟𝑟𝑜𝑟𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝐶𝑙𝑎𝑠𝑠1+𝑊𝑒𝑖𝑔ℎ𝑡2×𝐸𝑟𝑟𝑜𝑟𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝐶𝑙𝑎𝑠𝑠2+𝐶×𝐸𝑟𝑟𝑜𝑟𝑝𝑒𝑛𝑎𝑙𝑡𝑦
# The weights are selected so the theoretical maximum error in both classes is the same (see the help for the exact equation).
#
# Random State
# The random seed. Remember to use your student ID.
#
# Iterations
# The solution comes from an iterative model, thus we specify a maximum number of iterations. Remember to check for convergence after it has been solved!
#
# Solver
# Data science functions are complex ones, with thousands, millions, or even billions of parameters. Thus we need to use the best possible solver for our problems. Several are implemented in scikit-learn. The help states that:
# For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.
# For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss; ‘liblinear’ is limited to one-versus-rest schemes.
# ‘newton-cg’, ‘lbfgs’ and ‘sag’ only handle L2 penalty, whereas ‘liblinear’ and ‘saga’ handle L1 penalty.
# We will use 'saga', a very efficient solver. You can read all about it here.
#
# Warm start
# Scikit-learn allows for multiple adjustments to the training. For example, you can try first with a little bit of data just to check if everything is working, and then, if you set warm_start = True before, it will retrain starting from the original parameters. Allows for dynamic updating as well. warm_start = False means whenever we give it new data, it will start from scratch, forgetting what it previously learned.

