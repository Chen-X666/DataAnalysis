# _*_ coding: utf-8 _*_
"""
Time:     2021/6/17 18:02
Author:   ChenXin
Version:  V 0.1
File:     oneHot.py
skLearn :https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
Describe: Write during the internship at Hikvison, Github link: https://github.com/Chen-X666
"""
import pandas as pd
from sklearn.preprocessing import  OneHotEncoder

"""
data = pd
label = [] 需要one-hot的标签
"""
def oneHot(data,label):

    tempdata = data[label]
    enc = OneHotEncoder()
    enc.fit(tempdata)
    #one-hot编码的结果是比较奇怪的，最好是先转换成二维数组
    tempdata = enc.transform(tempdata).toarray()
    print('取值范围整数个数：',enc.get_feature_names())

    #再将二维数组转换为DataFrame，记得这里会变成多列
    tempdata = pd.DataFrame(tempdata,columns=[enc.get_feature_names()])

    for i in label:
        del data[i]
    newdata = data.join(tempdata)
    print(newdata)
    return newdata

if __name__ == '__main__':
    #测试数据集
    data = [['自有房',40,50000],
            ['有房',22,13000],
            ['自房',30,30000]]
    data = pd.DataFrame(data,columns=['house','age','income'])
    print(data)

    label = ['house','age']
    oneHot(data,label)


