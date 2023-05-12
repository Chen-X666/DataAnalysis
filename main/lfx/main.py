# _*_ coding: utf-8 _*_
"""
Time:     2022/2/26 1:47
Author:   ChenXin
Version:  V 0.1
File:     ExponentialSmoothing.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd

from dataPretreatment import SMOTE

if __name__ == '__main__':
    df = pd.read_csv('103.csv')
    #SMOTE样本均衡
    del df['A']
    df = pd.concat([df,df],axis=0)
    #df = pd.concat([df,df],axis=0)
    df.reset_index()
    print(df)
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    X,y = SMOTE.sample_balance(X,y)
    df = pd.concat([y,X],axis=1)
    print(df)
    df.to_csv('2.csv',index=False)
    print('数据增强中...')
    print('原有数据')