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
import dataReading.dataReading as dataReading
import dataPretreatment.SMOTE as SMOTE
import dataPretreatment.standardization as standardization
import dataReading.dataPicReading as dataPicReading
import seaborn as sns
import dataPretreatment.outlierDection as outlierDection
import dataPretreatment.Clustering as clustering
import dataAnalysisModel.regression as regression
import dataAnalysisModel.RandomForest as randomForest

if __name__ == '__main__':
    #数据预处理
    trainingData = pd.read_excel('typedata.xls', sheet_name='training')
    #查看数据状态
    dataReading.dataSimpleReading(trainingData)
    # columns = ['AGE','CUSTOMER_CLASS',
    #           'LINE_TENURE','SUBPLAN','SUBPLAN_PREVIOUS','NUM_TEL','NUM_ACT_TEL']
    # columns = ['GENDER', 'MARITAL_STATUS',
    #            'PAY_METD', 'PAY_METD_PREV', 'CUSTOMER_TYPE']
    #dataPicReading.dataBarReading(data=trainingData,columns=columns)
    #dataPicReading.dataBoxReading(data=trainingData,columns=columns,picWidth=3,picHigh=3)
    #转换为数值型数据   x
    trainingData['LINE_TENURE'] = pd.to_numeric(trainingData['LINE_TENURE'], errors='coerce')
    trainingData['AGE'] = pd.to_numeric(trainingData['AGE'], errors='coerce')
    # columns = ['GENDER', 'MARITAL_STATUS',
    #            'PAY_METD', 'PAY_METD_PREV', 'CUSTOMER_TYPE']
    # dataReading.dataStringCount(data=trainingData,columns=columns)
    #去重
    trainingData.drop_duplicates()
    #年龄分箱
    bins = [0,30,60,80]
    labels = [1,2,3]
    trainingData = dataBinning.dataBinning(trainingData, bins, column='AGE', labels=labels)
    #标签编码
    categoryPM, trainingData = labelEncoder.labelEncoder(trainingData, 'PAY_METD')
    categoryPMP, trainingData = labelEncoder.labelEncoder(trainingData, 'PAY_METD_PREV')
    categoryCT, trainingData = labelEncoder.labelEncoder(trainingData, 'CUSTOMER_TYPE')
    categoryG, trainingData = labelEncoder.labelEncoder(trainingData, 'GENDER')
    categoryM, trainingData = labelEncoder.labelEncoder(trainingData, 'MARITAL_STATUS')
    #查看数据格式
    print(trainingData.dtypes)
    #转换数据格式
    trainingData['LINE_TENURE'] = pd.to_numeric(trainingData['LINE_TENURE'], errors='coerce')
    trainingData['AGE'] = pd.to_numeric(trainingData['AGE'], errors='coerce')
    #去空白值
    trainingData.dropna(how='any', axis=0, inplace=True)
    #support字段的聚类
    # trainingData['SUBPLAN'] = clustering.kmeanClustering(data1=trainingData,columns='SUBPLAN',x='CUSTOMER_TYPE',y="SUBPLAN",num = 2)
    # trainingData['SUBPLAN_PREVIOUS'] = clustering.kmeanClustering(data1=trainingData, columns='SUBPLAN_PREVIOUS', x='CUSTOMER_TYPE',
    #                                                     y="SUBPLAN_PREVIOUS",num = 4)
    trainingData['LINE_TENURE'] = clustering.kmeanClustering(data1=trainingData, columns='LINE_TENURE',
                                                                  x='CUSTOMER_TYPE',
                                                                  y="LINE_TENURE",num=5)
    #
    #去异常值——————孤独森林
    trainingData = outlierDection.isolationForest(data=trainingData)
    print(trainingData)
    # 去空白值
    trainingData.dropna(how='any', axis=0, inplace=True)
    # 看样本是否均衡
    dataReading.label_samples_summary(trainingData)
    '''
    result:
    0    1754
    1     233
    '''
    #做柱状图
    #columns = ['AGE', 'GENDER', 'MARITAL_STATUS', 'CUSTOMER_CLASS', 'LINE_TENURE', 'SUBPLAN',
    #           'SUBPLAN_PREVIOUS', 'NUM_TEL', 'NUM_ACT_TEL', 'PAY_METD', 'PAY_METD_PREV', 'CUSTOMER_TYPE']
    #dataPicReading.dataHistogramReading(data=trainingData,columns=columns,picWidth=4,picHigh=3)
    # 相关性分析
    #columns = ['AGE', 'GENDER','MARITAL_STATUS','CUSTOMER_CLASS','LINE_TENURE', 'SUBPLAN',
    #'SUBPLAN_PREVIOUS', 'NUM_TEL', 'NUM_ACT_TEL','PAY_METD','PAY_METD_PREV','CUSTOMER_TYPE']
    #dataReading.relatedAnalysisReading(data=trainingData,columns=columns)
    '''
    result:
                NUM_TEL     NUM_ACT_TEL
     NUM_TEL        1           1
     NUM_ACT_TEL    1           1
    '''
    # 删掉NUM_ACT_TEL
    del trainingData['NUM_ACT_TEL']

    #x取除了最后一行，y取最后一行
    X = trainingData.iloc[:, :-1]
    y = trainingData.iloc[:, -1]
    #SMOTE做样本均衡
    X_resampled,y_resampled = SMOTE.sample_balance(X=X,y=y)
    print(X_resampled['AGE'])
    trainingData = X_resampled.join(y_resampled)
    print(trainingData)
    dataReading.label_samples_summary(trainingData)
    '''
    result:
    0    1754
    1    1754
    '''
    X = trainingData.iloc[:, :-1]
    y = trainingData.iloc[:, -1]
    #regression.regression(X,y)
    #trainingData.to_csv('1.csv', index=False)
    randomForest.decisionTree(X,y)



