# _*_ coding: utf-8 _*_
"""
Time:     2021/10/28 10:20
Author:   ChenXin
Version:  V 0.1
File:     bulletAnalysis.py
Describe:  Github link: https://github.com/Chen-X666
"""

import pandas as pd
from ExploratoryDataAnalysis import dataReview
import joblib

if __name__ == '__main__':
    trainingData = pd.read_csv('CandidateWord.csv',encoding='GBK')
    #.sort_values(by=['col2','col3'],ascending=False)
    #trainingData['Word'] =trainingData['Word'].str.len()
    del trainingData['Word']
    del trainingData['Frequence']
    # del trainingData['Tname']
    #del trainingData['Num']
    # del trainingData['View']
    # del trainingData['Length']
    #del trainingData['Danmaku']
    # del trainingData['Share']
    # del trainingData['Like']
    # del trainingData['Reply']
    # del trainingData['Coin']
    # del trainingData['Favorite']
    #del trainingData['Rep']
    #del trainingData['Den']
    #trainingData['Frequence'] = log(trainingData['Frequence'])
    trainingData = trainingData[trainingData['Mark'] >= 0]
    dataReading.dataSimpleReview(trainingData)
    #trainingData[]
    # 去离群点
    #trainingData = trainingData[trainingData['Frequence'] <= 36000]
    #trainingData['Frequence'] = standardization.ZScore(data=trainingData['Frequence'].to_list)
    #trainingData = trainingData[trainingData['W'] <= 6]
    #del trainingData['W']
    # 频数分箱
    # bins = [0,4000,8000,12000,16000,20000,24000,28000,32000,36000]
    # labels = [1,2,3,4,5,6,7,8,9]
    # trainingData = dataBinning.dataBinning(trainingData, bins, column='Frequence', labels=labels)
    X = trainingData.iloc[:, :-1]
    y = trainingData.iloc[:, -1]
    #离群点提取
    # print(np.array(X))
    # a = abNormalDetect.OneClassSvm(np.array(X))
    # np.savetxt("new.csv", a, delimiter=',')
    # print(a)
    # trainingData = pd.read_csv('标注集.csv', encoding='GBK')
    # trainingData['mark'] = 0
    # for i in a:
    #         trainingData.loc[(trainingData['Frequence'] == i[1]) & (trainingData['Mut'] == i[2]) & (trainingData['Freedom_L'] == i[3])  & (trainingData['Freedom_R'] == i[4]), 'mark'] = '1'
    # trainingData.to_csv('result.csv',encoding='GBK')
    # #SMOTE样本均衡
    #X,y = SMOTE.sample_balance(X,y)
    #普通样本均衡
    print(trainingData)
    ones_subset = trainingData.loc[trainingData["Mark"] == 1, :][0:3000]
    print(len(ones_subset))
    number_of_1s = len(ones_subset)
    zeros_subset = trainingData.loc[trainingData["Mark"] == 0, :]
    sampled_zeros = zeros_subset.sample(number_of_1s)
    clean_df = pd.concat([ones_subset, sampled_zeros], ignore_index=True)
    X = clean_df.iloc[:, :-1]
    y = clean_df.iloc[:, -1]
    #建立决策树模型
    #ExploratoryDataAnalysis.dataSimpleReading(X)
    print(X)
    print(y)
    print('开始构建模型')
    randomForest = RandomForest.decisionTree(X, y)
    joblib.dump(randomForest, 'bulletNewWordDiscoveryRandomForest.model')
    #logisticRegression.LogisticRegress(X,y)
    #knn.KNN(X,y)
    print('开始预测')
    # 模型预测
    # 预测集数据预处理
    # predict_data = pd.read_csv('预测集.csv',encoding='GBK')
    # del predict_data['Word']
    # del predict_data['Frequence']
    # #predict_data = predict_data[predict_data['Mark'] >= 0]
    # predict_data = predict_data.iloc[:, :-1]
    # print("输入特征数据：{}".format(predict_data.T))
    # print("模型预测结果：{}".format(randomForest.predict(predict_data)))
    # predictData = randomForest.predict(predict_data)
    # trainingData = pd.read_csv('预测集.csv', encoding='GBK')
    # #trainingData = trainingData[trainingData['Mark'] >= 0]
    # trainingData['Mark_Pre'] = predictData.tolist()
    # trainingData.to_csv('预测集结果.csv',encoding='GBK')