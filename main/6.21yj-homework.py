# _*_ coding: utf-8 _*_
"""
Time:     2021/6/21 9:52
Author:   ChenXin
Version:  V 0.1
File:     6.21yj-homework.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Chen-X666
"""
import pandas as pd
import numpy as np
import dataPretreatment.oneHot as oneHot
import dataPretreatment.dataBinning as dataBinning
import dataPretreatment.labelEncoder as labelEncoder

if __name__ == '__main__':
    data = pd.read_excel('typedata.xls')
    #年龄分箱
    bins = [0,30,60,80]
    data = dataBinning.dataBinning(data,bins,column='AGE')
    print(data)
    #标签编码
    categoryPM,data = labelEncoder.labelEncoder(data,'PAY_METD')
    categoryPMP, data = labelEncoder.labelEncoder(data, 'PAY_METD_PREV')
    categoryCT, data = labelEncoder.labelEncoder(data, 'CUSTOMER_TYPE')
    categoryG, data = labelEncoder.labelEncoder(data, 'GENDER')
    categoryM, data = labelEncoder.labelEncoder(data, 'MARITAL_STATUS')
    #两个值的相关系数研究 看看要不要删掉
    a = data[['NUM_TEL','NUM_ACT_TEL']].corr()
    #b = data[['SUBPLAN','SUBPLAN_PREVIOUS']].corr()
    print(a)
    #df = data.to_csv('1.csv')
    #oneHot.oneHot(data,)