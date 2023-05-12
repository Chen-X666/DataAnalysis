# _*_ coding: utf-8 _*_
"""
Time:     2021/6/22 21:50
Author:   ChenXin
Version:  V 0.1
File:     ExploratoryDataAnalysis.py
Describe: Github link: https://github.com/Chen-X666
"""
import matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import re
import seaborn as sns

matplotlib.use('TkAgg')
def dataSimpleReview(data):
    # 显示所有列 行
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(data)
    print('{:-^60}'.format('Data description'))
    print(df.head().append(df.tail()))
    print('{:-^60}'.format('Data basic statistic'))
    print(df.describe())
    print('{:-^60}'.format('Missing value detection'))
    print(data[data.duplicated(keep=False)])
    print('{:-^60}'.format('duplicate data detection'))
    data_dup = data[data.duplicated(keep="last")]
    print(data_dup.shape)
    print(data_dup)
    print('{:-^60}'.format('Data type checking'))
    print(df.info())
    print('{:-^60}'.format('Data skewness checking'))
    print(df.skew())
    print('{:-^60}'.format('Residual percentage'))
    print(((df.isnull().sum())/df.shape[0]).sort_values(ascending=False).map(lambda x:"{:.6%}".format(x)))
    print('{:-^60}'.format('Column missing value check'))
    print(df.isnull().any(axis=0).sum())
    print('{:-^60}'.format('Row mssing value check'))
    print(df.isnull().any(axis=1).sum())
    pd.reset_option("display.max_rows")  # restore to the initial set


def dataStringCount(data,columns):
    data = pd.DataFrame(data)
    print('{:-^60}'.format('String count'))
    for i in columns:
        print(data[i].value_counts())

# 类样本均衡审查
def label_samples_summary(df,column):
    '''
    查看每个类的样本量分布
    :param df: 数据框
    :return: 无
    '''
    print('{:*^60}'.format('Labesl samples count:'))
    print(df.groupby([column])[column].count())

# 相关性分析
def relatedAnalysis(data, columns):
    X_combine = pd.DataFrame(data[columns])
    print('{:*^60}'.format('相关系数分析:'))
    corr = X_combine.corr().round(2)
    print(X_combine.corr().round(2))  # 输出所有输入特征变量以及预测变量的相关性矩阵
    sns.heatmap(corr, cmap='Blues', annot=True)
    #sns.pairplot(data=corr,diag_kind='kde')#非常慢
    plt.show()

