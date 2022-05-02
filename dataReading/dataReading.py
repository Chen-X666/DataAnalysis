# _*_ coding: utf-8 _*_
"""
Time:     2021/6/22 21:50
Author:   ChenXin
Version:  V 0.1
File:     dataReading.py
Describe: Github link: https://github.com/Chen-X666
"""
import pandas as pd
from matplotlib import pyplot as plt
import re
import seaborn as sns
def dataSimpleReading(data):
    # 显示所有列 行
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(data)
    print('{:-^60}'.format('数据基本情况'))
    print(df.head().append(df.tail()))
    print('{:-^60}'.format('数据基本统计分析'))
    print(df.describe())
    print('{:-^60}'.format('数据类型查看'))
    print(df.info())
    print('{:-^60}'.format('数据峰度查看；如果训练集和测试集分布不一致，就要考虑进行分布转换'))

    print('{:-^60}'.format('列 空值查看'))
    print(df.isnull().any(axis=0).sum())
    print('{:-^60}'.format('行 空值查看'))
    print(df.isnull().any(axis=1).sum())
    pd.reset_option("display.max_rows")  # 恢复默认设置


def dataStringCount(data,columns):
    data = pd.DataFrame(data)
    print('{:-^60}'.format('字符串数据计数'))
    for i in columns:
        print(data[i].value_counts())

# 类样本均衡审查
def label_samples_summary(df):
    '''
    查看每个类的样本量分布
    :param df: 数据框
    :return: 无
    '''
    print('{:*^60}'.format('Labesl samples count:'))
    print(df.iloc[:, 0].groupby(df.iloc[:, -1]).count())

# 相关性分析
def relatedAnalysisReading(data,columns):
    X_combine = pd.DataFrame(data[columns])
    print('{:*^60}'.format('相关系数分析:'))
    print(X_combine.corr().round(2))  # 输出所有输入特征变量以及预测变量的相关性矩阵
    sns.pairplot(data=X_combine.corr().round(2),diag_kind='kde')
    sns.pairplot(data[columns])
    plt.show()
