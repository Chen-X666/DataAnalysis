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
    print('{:-^60}'.format('Random forest construction'))
    # sample and features
    n_samples, n_features = X_train.shape
    print('{:-^60}'.format('data check'))
    print('The number of data: {0} | The number of features: {1}'.format(n_samples, n_features))

    # using the grid-search to find the optimal model
    model_RF = RandomForestClassifier(random_state=0,class_weight='balanced',criterion='gini')
    parameters = {'n_estimators': range(10, 100, 10),
                  'max_depth': range(2, 18, 2),
                 #'min_samples_split': range(10, 500, 20),
                  'max_features': range(2, 6, 1)
                  }
    # parameters = {'n_estimators': [20],
    #               'max_depth': [6],
    #               # 'min_samples_split': range(10, 500, 20),
    #               'max_features': [5]
    #               }
    # use the grid-search and cross-validation to train the model
    model_gs = GridSearchCV(estimator=model_RF, param_grid=parameters, cv=10, n_jobs=-1,scoring='roc_auc')
    model_gs.fit(X_train, y_train)
    print('Optimal score:', model_gs.best_score_)
    print('Optimal parameters:', model_gs.best_params_)
    model_rf = model_gs.best_estimator_
    # use the optimal parameters to fit the model
    model_rf.fit(X_train, y_train)
    pre_y = model_rf.predict(X_test)
    y_score = model_rf.predict_proba(X_test)
    print(X_train.columns)
    # evaluate the model
    ROC(modelName='Random Forest',y_test=y_test, y_score=y_score)

    featureImportant(X_train, model_rf)

    ConfusionMatrix(y_test, pre_y)

    # accuracy，precision，recall，and f1
    valueEvaluation(y_test, pre_y, y_score)

    # plot learning curve
    plot_learning_curve(model_gs, X_train, X_test, y_train, y_test)

    # show the tree but need the third-part tools
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
    # import joblib
    # joblib.dump(model_rf, "my_random_forest.joblib")  # save

    return model_rf



def featureImportant(X,model):
    coef_lr = pd.DataFrame({'var': X.columns,
                            'coef': model.feature_importances_.flatten()
                            })

    index_sort = np.abs(coef_lr['coef']).sort_values().index
    coef_lr_sort = coef_lr.loc[index_sort, :]
    plt.bar(coef_lr_sort['var'], coef_lr_sort['coef'])
    plt.title('feature importance')
    plt.xlabel('features')
    plt.ylabel('importance')
    print((coef_lr_sort['coef']).to_list())
    plt.xticks(range(len((coef_lr_sort['var']).to_list())), (coef_lr_sort['var']).to_list(), rotation=10, size=8)
    # show the number
    for a, b in zip(coef_lr_sort['var'], coef_lr_sort['coef']):
        plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=8)
    plt.show()
    print(coef_lr_sort)


def confusionMatrixTable(y_test,pre_y):
    TN, FP, FN, TP = confusion_matrix(y_test, pre_y).ravel()  # 获得混淆矩阵并用ravel将四个值拆开赋予TN,FP,FN,TP
    confusion_matrix_table = prettytable.PrettyTable(['', 'predict-0', 'predict-'])
    confusion_matrix_table.add_row(['real-0', TP, FN])
    confusion_matrix_table.add_row(['real-1', FP, TN])
    print('{:-^60}'.format('Confused Matrix'), '\n', confusion_matrix_table)
