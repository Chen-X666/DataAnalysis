# _*_ coding: utf-8 _*_
"""
Time:     2022/3/28 16:53
Author:   ChenXin
Version:  V 0.1
File:     model.py
Describe:  Github link: https://github.com/Chen-X666
"""
# !pip install autots
from autots import auto_timeseries
import pandas as pd

df = pd.read_csv("./data/data.csv", usecols=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
train_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Train')
test_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Test')
plt.legend()
plt.grid()
plt.show()
model = auto_timeseries(forecast_period=219, score_type='rmse', time_interval='D', model_type='best')
model.fit(traindata= train_df, ts_column="Date", target="Close")
future_predictions = model.predict(testdata=219)
