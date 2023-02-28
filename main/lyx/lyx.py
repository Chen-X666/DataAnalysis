# _*_ coding: utf-8 _*_
"""
Time:     2022/6/24 11:01
Author:   ChenXin
Version:  V 0.1
File:     lyx.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
#读取数据
df = pd.read_csv('数据分析与实践期末/customdata.csv',encoding='utf-8')
df.info()
print(df)
#1. 年龄处理
##1.1 删除小数点
# df = df [df['AGE'] % 1 == 0]
# print(df)
##1.2 小数点四舍五入
# df['AGE'] = df['AGE'].map(lambda x: round(x,0))
# print(df)
#2. 性别处理
##2.1 去掉O
# print(df)
# df = df[df['GENDER'] != 'O']
# print(df)
#3. tenure处理
#3.1 强制类型转换
#去空值
# df = df[df['LINE_TENURE'] != ' ']
# df['LINE_TENURE'] = df['LINE_TENURE'].map(lambda x: int(x))
# df.info()
