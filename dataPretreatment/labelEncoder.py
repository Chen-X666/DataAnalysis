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
def labelEncoder(data,label):
    #
    # tempdata = data[label]
    # tempdata = np.array(tempdata)
    # label_encoder = LabelEncoder()
    # label_encoder.fit(tempdata)
    # print(label_encoder.classes_)   # 对原始编码去重，相当于np.array(set(data))
    #
    # integer_encoded = label_encoder.transform(tempdata)     # 对原数据编码
    # print(integer_encoded)
    # data = label_encoder.transform(tempdata) #
    category = pd.Categorical(data[label])
    data[label] = pd.Categorical(data[label]).codes
    return category,data

if __name__ == '__main__':
    #测试数据集
    data = [['自有房',40,50000],
            ['房',22,13000],
            ['房',30,30000]]
    data = pd.DataFrame(data,columns=['house','age','income'])
    print(data)

    label = ['house']
    labelEncoder(data,label)


