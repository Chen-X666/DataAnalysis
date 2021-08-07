# _*_ coding: utf-8 _*_
"""
Time:     2021/8/7 13:42
Author:   ChenXin
Version:  V 0.1
File:     Pandas.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
import numpy as np

#基本操作
def basicProcess():
    df=pd.Dataframe(columns=[],index=[],data=[]) ##创建一个Dataframe
    df.head(n=10) ## 显示前n行的数据
    df.tail(n=10) ## 显示尾n行的数据
    df.index      ##返回df的行索引值，是一个array
    df.columns ##返回df的列名，是一个array
    df.values    ##返回df的所有值，是一个2D array
    df.describe('all') ##统计每列的min, max,mean,std,quantile
    df.dtypes  ##返回每列数据的类型
    df.T  ##转置数据框
    df.sort_index(axis=1,ascending=False) ##按索引排序所有列，升序或者降序

#切片操作
def cutTable(df,index):
    df['column_name']  ##利用列名直接取某一列
    df[df.columns[index]]  ##适合于你不知道列名，但是知道它在第几列

    df.loc[index]  ##根据行的位置，取特定行数据（列全取）
    df.loc[[index], ['a', 'b']]  ##取index行的，ab两列数据
    df.loc[[index], 'a':'b']  ##取index行的，列名为'a' 到 列名为 'b'的所有列
    ##总之，列不能通过索引来取数

    df.iloc[0:10, 0:10]  ##切片后面的值取不到，即col_index=10,row_indx=10取不到
    df.iloc[[0, 5, 10], [1, 8, 10]]  ##可按照需求，选择特定的行和列
    ##总之iloc之内的数据都是数字，不能是行名列名

    df[df.A > 0]  ##取出A列中大于0的数，
    df[df['A'].isin(['one', 'two'])]
    ##取出A列中包含'one','two'的数据,这个功能很强大，##可以帮助我们filter出符合条件的数据。

    df['A'] = np.array([1] * len(df))  ##用数组给某列赋值
    df.loc[:, ['a', 'c']] = []  ##根据位置赋值
    ##知道如何取数，就能轻松给数据框赋值啦。

def orderColumn(df):
    df.sort_values(by='column_Name', ascending=True)  ##按某列升序排序
    df.sort_index(axis=1, ascending=True)  ##索引排序

def mergeTable(df,df1,df2):
    pd.concat([df1[:], df2[:], ...], axis=0)  ##按列拼接数据，要求列数和列名一样
    pd.concat([df1, df2, ...], axis=1) ##按行拼接数据，行数和行索引相同
    ##如果数据结构不一样，可以选择join="inner","outer",..sql中的操作

    df.append(df1[:], ignore_index=True)  ##将会重新设定index

    df.merge(df1, on=['column_name', ...], how='inner')  ##内联表，根据主键来拼接
    ##how = "inner", "left", "right", "outer"分别表示内连接，左连接，右连接，外连接。
    ##具体如何连接，大家去温习一下sql中的表连接操作吧.

def groupColum(df,key,key1,key2):
    def function(x):
        print()
    grouped = df.groupby(key)  ##将某个主键按照类别分组，默认是列主键
    grouped = df.groupby(key, axis=1)  ##按照某个key分组，行操作
    grouped = df.groupby([key1, key2, ...])  ##可以依次group多个key。
    grouped.groups  ##返回分组的结果
    grouped.get_group('a')  ## 选择其中一个分组的类别，查看该类别的数据

    grouped.aggregate(np.sum)  ##分组求和，常见操作
    grouped.size()  ##分组统计数量
    grouped.describe()  ##分组查看描述统计结果

    grouped.agg([np.sum, np.std, np.mean])  ##同时求和，均值方差。
    grouped.apply(lambda x: function(x))  ##可以接上apply函数，进行自定义操作

    grouped.filter(lambda x: len(x) > 2, dropna=True)  ##类似这种filter操作
    ##根据自己需求，都能够相应地实现。

def outputSetting():
    pd.set_option("display.height", 200)  ##设置显示结果的高度
    pd.set_option("display.max_seq_items", 200)  ##设置序列显示的最大个数
    pd.set_option("display.max_columns", 120)  ##设置数据框显示的列数
    pd.set_option("display.max_rows", 50)  ##设置数据框显示的行数

def saveTable(df,file_path,index_col,startcol,startrow,header):
    ##先是读取数据
    pd.read_csv(file_path, header=header, sep=',', index_col=index_col)  ##常用的取数据函数
    pd.read_excel(file_path, sheetName='sheetName', startcol=startcol, startrow=startrow, header=header)
    ##保存数据相对规范的话是如下代码：
    writer = pd.ExcelWriter('excel name.xlsx')  ##新建一个excel
    df.to_excel(writer, sheetName='dfName', startcol=startcol,startrow=startrow)
    writer.save()