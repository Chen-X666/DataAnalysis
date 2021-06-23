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
def dataSimpleReading(data):
    # 显示所有列 行
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame(data)
    print('{:-^60}'.format('数据基本统计分析'))
    describe = df.describe()
    print(describe)
    print('{:-^60}'.format('数据类型查看'))
    print(df.dtypes)
    print('{:-^60}'.format('空值查看'))
    print(df.isnull().sum())
    pd.reset_option("display.max_rows")  # 恢复默认设置

# 类样本均衡审查
def label_samples_summary(df):
    '''
    查看每个类的样本量分布
    :param df: 数据框
    :return: 无
    '''
    print('{:*^60}'.format('Labesl samples count:'))
    print(df.iloc[:, 0].groupby(df.iloc[:, -1]).count())

def dataBoxReading(data,columu):
    '''
    箱型图构建
    :param data: 数据框
    :return: 无
    '''
    df = pd.DataFrame(data)
    box_1 = df[columu]
    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('Data of boxplot', fontsize=20)  # 标题，并设定字号大小
    plt.boxplot(box_1)  # grid=False：代表不显示背景中的网格线
    # data.boxplot()#画箱型图的另一种方法，参数较少，而且只接受dataframe，不常用
    plt.show()  # 显示图像

def relatedAnalysisReading(data):
    # print('{:-^60}'.format('相关性分析'))
    # short_name = ['AGE', 'GENDER', "LINE_TENURE ", "SUBPLAN", "SUBPLAN_PREVIOUS", "NUM_TEL", "NUM_ACT_TEL"]
    # long_name = train_data.columns
    # name_dict = dict(zip(long_name, short_name))  # 组成字典
    # print(name_dict, '\n')
    # print(train_data.iloc[:-3, :].corr().round(2).rename(index=name_dict, columns=name_dict))
    return 0
