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
import dataReading.dataReading as dataSimpleReading

if __name__ == '__main__':
    #数据预处理
    data = pd.read_excel('typedata.xls')
    #查看数据状态
    dataSimpleReading.dataSimpleReading(data)
    #去重
    data.drop_duplicates()
    #年龄分箱
    bins = [0,30,60,80]
    labels = [1,2,3]
    data = dataBinning.dataBinning(data,bins,column='AGE',labels=labels)
    print(data)
    #标签编码
    categoryPM,data = labelEncoder.labelEncoder(data,'PAY_METD')
    categoryPMP, data = labelEncoder.labelEncoder(data, 'PAY_METD_PREV')
    categoryCT, data = labelEncoder.labelEncoder(data, 'CUSTOMER_TYPE')
    categoryG, data = labelEncoder.labelEncoder(data, 'GENDER')
    categoryM, data = labelEncoder.labelEncoder(data, 'MARITAL_STATUS')
    #两个值的相关系数研究 看看要不要删掉
    # a = data[['NUM_TEL','NUM_ACT_TEL']].corr()
    # b = data[['SUBPLAN','SUBPLAN_PREVIOUS']].corr()
    # print(a)
    print(data.dtypes)
    data['LINE_TENURE'] = pd.to_numeric(data['LINE_TENURE'], errors='coerce')
    print(dataSimpleReading.dataSimpleReading(data))
    #去空白值
    data.dropna(how='any', axis=0, inplace=True)
    #存
    df = data.to_csv('1.csv',index=False)
    print(data.dtypes)
    #data = data.infer_objects()


