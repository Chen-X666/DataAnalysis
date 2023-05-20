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
    print('='*20+"Logic Regression Model Construction"+'='*20)
    # 训练集和验证集4：1切分
    #X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=0)

    #See the sample and features
    n_samples,n_features=X_train.shape
    print('{:-^60}'.format('Check Data'))
    print('The number of the data: {0} | The features of the data: {1}'.format(n_samples,n_features))

    # train the logistic regression model
    parameters = {'penalty':['l1','l2'],
                  'C':[0.001,0.01,0.1,1,10,100],
                  'solver':['liblinear','saga','lbfgs']} #
    # parameters = {'penalty': ['l2'], 'C': [100], 'solver': ['lbfgs']}  # 最优参数
    # train the cross-validation and grid search model
    model_t = GridSearchCV(estimator=linear_model.LogisticRegression(max_iter=10000), param_grid=parameters,verbose=0,cv=10,n_jobs=-1,scoring='roc_auc')  # 建立交叉检验模型对象，并行数与CPU一致
    model_t.fit(X_train, y_train)
    print('{:-^60}'.format('The optimal model parameters and score'))
    print('The best score:', model_t.best_score_)
    print('The best parameters:', model_t.best_params_)
    model = model_t.best_estimator_
    # fit using the optimal parameters
    model.fit(X_train, y_train)

    # predict
    y_predict = model.predict(X_test)  #
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'], index=y_test.index)
    y_test_predict_df = pd.concat([y_test, y_predict_df], axis=1)
    y_score = model.predict_proba(X_test)  # 获得决策树对每个样本点的预测概率
    print('Test data VS. Real data', '-' * 30, '\n', y_test_predict_df)


    # initial evaluation
    print('Model','-'*30,'\n',model)
    print('Optimal model coefficient','-'*30,'\n',model.coef_)
    print('Optimal model intercept','-'*30,'\n',model.intercept_)
    print('Optimal model score','-'*30,'\n',model.score(X_test,y_test))

    print(y_test)
    print(y_score)
    # # evaluate model
    # ROC(modelName='Logistic Regression ROC',y_test=y_test, y_score=y_score)
    # # featureImportant(X_train,model)
    # ConfusionMatrix(y_test, y_predict)
    # # accuracy，precision，recall，and f1.
    valueEvaluation(y_test,y_predict,y_score)
    # # learning line
    # plot_learning_curve(model, X_train, X_test, y_train, y_test)
    return model

def featureImportant(X,model):
    sns.set()
    coef_lr = pd.DataFrame({'var': X.columns,
                            'coef': model.coef_.flatten()
                            })
    index_sort = np.abs(coef_lr['coef']).sort_values().index
    coef_lr_sort = coef_lr.loc[index_sort, :]
    plt.bar(coef_lr_sort['var'], coef_lr_sort['coef'])
    plt.title('feature importance')
    plt.xlabel('features')
    plt.ylabel('importance')
    print((coef_lr_sort['coef']).to_list())
    plt.xticks(range(len((coef_lr_sort['var']).to_list())),(coef_lr_sort['var']).to_list(), rotation=10, size=8)
    # plot the number
    for a, b in zip(coef_lr_sort['var'], coef_lr_sort['coef']):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=8)
    plt.show()  # show the figure
    print(coef_lr_sort)
# Parameter select regulation
# Penalty
# 'l1' penalty refers to LASSO regression (more strict), and 'l2' to Ridge regression (less strict, but easiear to solve). My advice: Start with LASSO, if it doesn't work or you are not happy with the results, move to Ridge. Large problems (over 10 million points) might require the use of Ridge regression in home computers.
# Penalty constant
# This refers to how much to weight the error in prediction versus the regularization (penalty) error. When optimizing the parameters, a penalization constant will try to optimise the following:
# 𝐸𝑟𝑟𝑜𝑟=𝐸𝑟𝑟𝑜𝑟𝑝𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛+𝐶×𝐸𝑟𝑟𝑜𝑟𝑝𝑒𝑛𝑎𝑙𝑡𝑦
# So the 𝐶 constant will balance both objectives. This is a tunable parameter, meaning we need to search for what leaves us happy. In general, if you think the model is being too strict, then reduce C; if it is being too lax, increase it.
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

