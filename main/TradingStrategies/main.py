# _*_ coding: utf-8 _*_
"""
Time:     2022/2/18 10:33
Author:   ChenXin
Version:  V 0.1
File:     ExponentialSmoothing.py
Describe:  Github link: https://github.com/Chen-X666
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import  pandas as pd
import  os
from keras.models import Sequential, load_model

import warnings                                  # 勿扰模式

from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

def create_dataset(dataset, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX),np.array(dataY)



if __name__ == '__main__':
    BCHAIN_MKPRU = pd.read_csv(r'C:\Users\Chen\Desktop\dataAnalysisPlatform\main\TradingStrategies\data\BCHAIN-MKPRU.csv', index_col=['Date'], parse_dates=['Date'])
    LBMA_GOLD = pd.read_csv(r'C:\Users\Chen\Desktop\dataAnalysisPlatform\main\TradingStrategies\data\LBMA-GOLD.csv', index_col=['Date'], parse_dates=['Date'])
    dataset = BCHAIN_MKPRU.values
    # 将整型变为float
    dataset = dataset.astype('float32')
    # 归一化 在下一步会讲解
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    train_size = int(len(dataset) * 0.65)
    trainlist = dataset[:train_size]
    testlist = dataset[train_size:]
    # 训练数据太少 look_back并不能过大
    look_back = 1
    trainX, trainY = create_dataset(trainlist, look_back)
    testX, testY = create_dataset(testlist, look_back)
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    model.save(os.path.join("DATA", "Test" + ".h5"))
    # make predictions

    #model = load_model(os.path.join("DATA","Test" + ".h5"))
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # 反归一化
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)
    # plt.plot(trainY)
    # plt.plot(trainPredict[1:])
    # plt.show()
    plt.plot(testY)
    plt.plot(testPredict[1:])
    plt.show()
