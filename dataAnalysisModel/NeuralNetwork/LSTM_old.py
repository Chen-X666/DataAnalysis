import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Bidirectional, RNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
import time
import os
import yfinance as yf
# 数据预处理及切分
BCHAIN_MKPRU = pd.read_csv(r'BCHAIN-MKPRU.csv', index_col=['Date'], parse_dates=['Date'])
LBMA_GOLD = pd.read_csv(r'LBMA-GOLD.csv', index_col=['Date'], parse_dates=['Date']).dropna()
dataset = BCHAIN_MKPRU.values
print("BCHAIN_MKPRU DATA:{}".format(BCHAIN_MKPRU.head()))
print("LBMA_GOLD DATA:{}".format(LBMA_GOLD.head()))

plt.figure(figsize=(10, 6))
plt.title("Time series chart of BCHAIN_MKPRU")
BCHAIN_MKPRU['Value'].plot();

plt.figure(figsize=(10, 6))
plt.title("Time series chart of LBMA_GOLD")
LBMA_GOLD['USD (PM)'].plot();

print("BCHAIN_MKPRU DATA Description")
BCHAIN_MKPRU.describe().transpose()
print("LBMA_GOLD DATA Description")
LBMA_GOLD.describe().transpose()

# 将整型变为float
dataset = dataset.astype('float32')
# 归一化
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset.reshape(-1,1))

train_size = int(len(dataset) * 0.7)
trainlist = dataset[:train_size]
testlist = dataset[train_size:]

# dataset 反归一化
dataset = scaler.inverse_transform(dataset)
dataset

# LSTM
best_model_mse = 1
best_look_back = 0
test_model_mse_list=[]
def create_dataset(data, look_back):
#这里的look_back与timestep相同
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back)]
        dataX.append(a)
        dataY.append(data[i + look_back])
    return np.array(dataX),np.array(dataY)

def loss(information_loss):
      # plt.figure(figsize=(12, 8))
      plt.plot(information_loss)
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.show()

for i in range(1,6):
  # 训练数据太少 look_back并不能过大
  look_back = i
  trainX, trainY = create_dataset(trainlist, look_back)
  testX, testY = create_dataset(testlist, look_back)
  trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
  testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

  # create and fit the LSTM network
  LSTM_model = Sequential()
  LSTM_model.add(LSTM(4, input_shape=(None, 1)))
  LSTM_model.add(Dense(1))
  LSTM_model.summary()
  LSTM_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
  LSTM_model_information = LSTM_model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
  LSTM_model_mse = LSTM_model.evaluate(testX,testY)[0]
  print("LSTM_model_mse: {}".format(LSTM_model_mse))
  test_model_mse_list.append(LSTM_model_mse)
  if(LSTM_model_mse < best_model_mse):
    best_model_mse = LSTM_model_mse
    best_look_back = i
  # model.save(os.path.join("DATA", "Test" + ".h5"))
  # make predictions

  # model = load_model(os.path.join("DATA","Test" + ".h5"))
  trainPredict = LSTM_model.predict(trainX)
  testPredict = LSTM_model.predict(testX)

  # 反归一化
  trainPredict = scaler.inverse_transform(trainPredict)
  trainY = scaler.inverse_transform(trainY)
  testPredict = scaler.inverse_transform(testPredict)
  testY = scaler.inverse_transform(testY)

  plt.plot(testY,label='true', lineWidth=1)
  plt.plot(testPredict[1:],label='prediction', lineWidth=1)
  plt.ylabel('value')
  plt.xlabel('date')
  plt.legend()
  plt.title("look_back="+str(i))
  plt.show()

  # shift train predictions for plotting
  trainPredictPlot = np.empty_like(dataset)
  trainPredictPlot[:, :] = np.nan
  trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

  # shift test predictions for plotting
  testPredictPlot = np.empty_like(dataset)
  testPredictPlot[:, :] = np.nan
  testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

  # plot baseline and predictions
  plt.plot(dataset,label='dataset', lineWidth=1)
  plt.plot(trainPredictPlot,label='train', lineWidth=1)
  plt.plot(testPredictPlot,label='test', lineWidth=1)
  plt.ylabel('value')
  plt.xlabel('date')
  plt.legend()
  plt.show()

  #模型损失可视化
  information_loss=LSTM_model_information.history['loss']  #模型训练损失
  loss(information_loss)
print("when look_back is {}, the model get the best mse score: {}".format(best_look_back, best_model_mse))
test_model_mse_line=test_model_mse_list
x=range(1,6,1)
plt.plot(x,test_model_mse_list,label='test_model_mse',lineWidth=1)
plt.ylabel('mse score')
plt.xlabel('look_back')
plt.legend()
plt.show()