# _*_ coding: utf-8 _*_
"""
Time:     2021/10/31 17:13
Author:   ChenXin
Version:  V 0.1
File:     processPredictData.py
Describe:  Github link: https://github.com/Chen-X666
"""
from dataPretreatment import dataBinning
import pandas as pd
if __name__ == '__main__':
    predict_data = pd.read_csv('预测集.csv',encoding='GBK')
    predict_data = predict_data[predict_data['Frequence'] <= 36000 ]
    bins = [0, 4000, 8000, 12000,16000,20000,24000,28000,34000,36000]
    labels = [1,2,3,4,5,6,7,8,9]
    predict_data = dataBinning.dataBinning(predict_data, bins, column='Frequence', labels=labels)
    print(predict_data)
    predict_data.to_csv('预测集分箱.csv',encoding='GBK',index_label=False)