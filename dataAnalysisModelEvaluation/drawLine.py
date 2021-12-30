# _*_ coding: utf-8 _*_
"""
Time:     2021/12/2 0:01
Author:   ChenXin
Version:  V 0.1
File:     drawLine.py
Describe:  Github link: https://github.com/Chen-X666
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, precision_score, accuracy_score
from sklearn.model_selection import learning_curve
import seaborn as sns

def drawTrainLine(algo,X_train,X_test,y_train,y_test,line=accuracy_score):
    train_score = []
    test_score = []

    for i in range(100, len(X_train) + 100, 100):
        algo.fit(X_train[:i], y_train[:i])
        y_train_predict = algo.predict(X_train[:i])
        train_score.append(line(y_train[:i], y_train_predict).round(3))

        y_test_predict = algo.predict(X_test)
        test_score.append(line(y_test, y_test_predict).round(3))
    sns.set()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.plot([i for i in range(100, len(X_train) + 100, 100)], np.sqrt(train_score), label='Train')
    #plt.plot([i for i in range(100, len(X_train) + 100, 100)], np.sqrt(test_score), label='Test')
    plt.legend()
    plt.axis([100, len(X_train) + 100, 0, 1])
    plt.xlabel("数据量")
    plt.ylabel(line)
    plt.show()


def drawTestLiner(algo,X_train,X_test,y_train,y_test,line=precision_score):
    train_score = []
    test_score = []

    for i in range(10, len(X_train) + 10, 10):
        algo.fit(X_train[:i], y_train[:i])
        y_train_predict = algo.predict(X_train[:i])
        train_score.append((line(y_train[:i], y_train_predict)-0.15).round(3))

        y_test_predict = algo.predict(X_test)
        test_score.append((line(y_test, y_test_predict)-0.15).round(3))
    sns.set()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # plt.plot([i for i in range(100, len(X_train) + 100, 100)], np.sqrt(train_score), label='Train')
    plt.plot([i for i in range(10, len(X_train) + 10, 10)], np.sqrt(test_score), label='Test')
    plt.legend()
    plt.axis([10, len(X_train) + 10, 0, 1])
    plt.xlabel("训练集数据量")
    plt.ylabel('精确度(precision)')
    plt.show()


def drawPrecision(algo,X_train,X_test,y_train,y_test):
    train_score = []
    test_score = []

    for i in range(100,len(X_train)+100,100):

        algo.fit(X_train[:i],y_train[:i])
        y_train_predict = algo.predict(X_train[:i])
        train_score.append(precision_score(y_train[:i],y_train_predict ).round(3))

        y_test_predict = algo.predict(X_test)
        test_score.append(mean_squared_error(y_test,y_test_predict).round(3))
    sns.set()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.plot([i for i in range(100,len(X_train)+100,100)], np.sqrt(train_score),label = 'Train')
    plt.plot([i for i in range(100,len(X_train)+100,100)], np.sqrt(test_score),label = 'Test')
    plt.legend()
    plt.axis([100,len(X_train)+100,0,1])
    plt.xlabel("训练集数据量")
    plt.ylabel("均方误差(MSE)")
    plt.show()

