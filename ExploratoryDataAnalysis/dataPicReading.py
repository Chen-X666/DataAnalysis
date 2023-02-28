# _*_ coding: utf-8 _*_
"""
Time:     2021/6/24 1:02
Author:   ChenXin
Version:  V 0.1
File:     dataPicReading.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()

def dataHistogramReading(data,columns,picWidth=2,picHigh=2):
    #设置风格
    if len(columns)>4 and picHigh<=2 and picWidth<=2 : return print('超出默认图数4,请定义width与high')
    fig, ax_arr = plt.subplots(picHigh, picWidth, figsize=(100, 50))
    #绘制多子图
    i,j = 0,0
    for column in columns:
        if j < picWidth:
            sns.distplot(data[column],ax=ax_arr[i][j])
            j = j +1
        else:
            i = i + 1
            j = 0
            sns.distplot(data[column], ax=ax_arr[i][j])
            j = 1
    plt.show()

#主要为了处理分类数据
def dataBarReading(data,columns,picWidth=2,picHigh=2,y_column_name=''):
    #设置风格
    if len(columns)>4 and picHigh<=2 and picWidth<=2 : return print('超出默认图数4,请定义width与high')
    fig, ax_arr = plt.subplots(picHigh, picWidth, figsize=(100, 50))
    #绘制多子图
    i,j = 0,0
    for column in columns:
        if j < picWidth:
            ax_arr[i][j].set_xlabel(column)
            sns.barplot(x=y_column_name,y=column,data=data,ax=ax_arr[i][j])
            j = j +1
        else:
            i = i + 1
            j = 0
            ax_arr[i][j].set_xlabel(column)
            sns.barplot(x=y_column_name,y=column,data=data,ax=ax_arr[i][j])
            j = 1
    plt.show()


# reference: https://cloud.tencent.com/developer/article/1506466
def dataScatterPlt(data,columns,pic,picWidth=2,picHigh=2):
    if len(columns)>4 and picHigh<=2 and picWidth<=2: return print('超出默认图数4,请定义width与high')
    fig, ax_arr = plt.subplots(picHigh, picWidth, figsize=(10, 5))
    #绘制多子图
    i,j = 0,0
    for column in columns:
        if j < picWidth:
            ax_arr[i][j].set_xlabel(column)
            sns.scatterplot(data[column],ax=ax_arr[i][j])
            ax_arr[i][j].set_xticks([])
            j = j +1
        else:
            i = i + 1
            j = 0
            # 设置坐标轴
            ax_arr[i][j].set_xlabel(column)
            sns.scatterplot(data[column], ax=ax_arr[i][j])
            # 去除坐标轴刻度
            ax_arr[i][j].set_xticks([])

            j = 1
    plt.show()


def dataBoxReading(data,columns,picWidth=2,picHigh=2):
    '''
    箱型图构建
    :param
    data: 数据框
    picWidth: 宽度放几个图
    :return: 无
    '''
    if len(columns)>4 and picHigh<=2 and picWidth<=2: return print('超出默认图数4,请定义width与high')
    fig, ax_arr = plt.subplots(picHigh, picWidth, figsize=(10, 5))
    #绘制多子图
    i,j = 0,0
    for column in columns:
        if j < picWidth:
            ax_arr[i][j].set_xlabel(column)
            sns.boxplot(data[column],ax=ax_arr[i][j])
            ax_arr[i][j].set_xticks([])
            j = j +1
        else:
            i = i + 1
            j = 0
            # 设置坐标轴
            ax_arr[i][j].set_xlabel(column)
            sns.boxplot(data[column], ax=ax_arr[i][j])
            # 去除坐标轴刻度
            ax_arr[i][j].set_xticks([])

            j = 1
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('1.csv')
    column = ['LINE_TENURE','SUBPLAN','SUBPLAN_PREVIOUS']
    #dataBoxReading(df,column)
    # 设置样式风格
    sns.set(style="darkgrid")
    # 构建数据
    tips = sns.load_dataset("tips")
    print(tips)
    """
    案例1：
    指定x分类变量进行分组，指定 y为数据分布，绘制垂直条形图
    """
    sns.barplot(x="day", y="total_bill", data=tips)

