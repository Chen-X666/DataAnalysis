# _*_ coding: utf-8 _*_
"""
Time:     2022/10/15 14:31
Author:   ChenXin
Version:  V 0.1
File:     XGBoost.py
Describe:  Github link: https://github.com/Chen-X666
"""
from prettytable import prettytable
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score as CSV, train_test_split, GridSearchCV  # k折 交叉验证
from sklearn.metrics import mean_squared_error as MSE, roc_curve, auc, accuracy_score, precision_score, recall_score, \
    f1_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dataAnalysisModel.classification.classificationMetrics import ROC, ConfusionMatrix, valueEvaluation


def XGboost(X_train, X_test, y_train, y_test):
    print('{:-^60}'.format('XGBoost construction'))
    # sample and features
    n_samples, n_features = X_train.shape
    print('{:-^60}'.format('data check'))
    print('The number of data: {0} | The number of features: {1}'.format(n_samples, n_features))

    # using the grid-search to find the optimal model
    model_XGBR = xgb.XGBClassifier()
    parameters = {'n_estimators': range(10, 100, 10),  #
                  'max_depth': range(2, 18, 2),
                  # 'min_samples_split': range(10, 500, 20),
                  # 'max_features': range(2, 6, 1)
                  }
    model_gs = GridSearchCV(estimator=model_XGBR, param_grid=parameters, cv=10, n_jobs=-1,
                            scoring='roc_auc')  #

    model_gs.fit(X_train, y_train)
    print('Optimal score:', model_gs.best_score_)
    print('Optimal parameters:', model_gs.best_params_)
    model_xgbr = model_gs.best_estimator_
    # use the optimal parameters to fit the model
    model_xgbr.fit(X_train, y_train)
    pre_y = model_xgbr.predict(X_test)
    y_score = model_xgbr.predict_proba(X_test)
    print(X_train.columns)
    # evaluate the model
    ROC(modelName='XGBoost', y_test=y_test, y_score=y_score)

    ConfusionMatrix(y_test, pre_y)

    # accuracy，precision，recall，and f1
    valueEvaluation(y_test, pre_y, y_score)

    # plot learning curve
    # plot_learning_curve(model_gs, X_train, X_test, y_train, y_test)
    return model_xgbr