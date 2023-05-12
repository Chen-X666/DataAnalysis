# _*_ coding: utf-8 _*_
"""
Time:     2021/6/18 0:49
Author:   ChenXin
Version:  V 0.1
File:     labelEncoder.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Chen-X666
"""
import copy
import pandas as pd
from sklearn.preprocessing import LabelEncoder  # 字符串转数值
import numpy as np  # numpy库
# 字符串分类转数值分类
# label encode标签编码(0, n-1)
def labelEncoder(df, labels):
    # Create a dictionary to store the LabelEncoder objects and their corresponding original labels
    encoders = {}

    # Apply label encoding to each categorical column in the dataframe
    for col in df[labels]:
        le = LabelEncoder()
        encoded_col = le.fit_transform(df[col])
        df[col] = encoded_col
        encoders[col] = {'encoder': le, 'classes': list(le.classes_)}

    # Print the original labels corresponding to each encoded value
    for col in encoders:
        print(f"Column '{col}': {encoders[col]['classes']} : {np.arange(0,len(encoders[col]['classes']),1)}")
    return df

if __name__ == '__main__':
    #测试数据集
    data = [['自有房',40,50000],
            ['房',22,13000],
            ['房',30,30000]]
    data = pd.DataFrame(data,columns=['house','age','income'])
    print(data)
    label = ['house','income']
    data = labelEncoder(data,label)
    print(data)



