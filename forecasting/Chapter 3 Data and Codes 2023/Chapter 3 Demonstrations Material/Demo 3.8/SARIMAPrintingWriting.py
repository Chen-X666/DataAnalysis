# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:31:22 2020

@author: abz1e14
"""
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas import read_excel
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  
plt.style.use('fivethirtyeight')

###############################
df = read_excel('PrintingWriting.xls', sheet_name='Data2', header=0, 
               index_col=0, parse_dates=True, squeeze=True)

df.plot()
# ARIMA model with (p, d, q)=(1, 1, 1)
mod = sm.tsa.statespace.SARIMAX(df, trend='c', order=(1,1,1))
#mod = sm.tsa.statespace.SARIMAX(df, order=(1,1,1),
                                #seasonal_order=(1,1,1,12))
results = mod.fit(disp=False)
print(results.summary())

# graphical statistics of model (correlogram = ACF plot)
results.plot_diagnostics(figsize=(15, 12))
plt.show()
#
# #============================================
# this code requires the fitted forecasts (for accuracy evaluation) to start 01 Jan 1979.
pred = results.get_prediction(start=pd.to_datetime('1972-01-01'), dynamic=False)
pred_ci = pred.conf_int()

print(pred_ci)

# this code requires the whole plot to start in 1956 (start year of data)

ax = df.plot(label='Original data')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.legend()
plt.show()

#=============================================
# MSE evaluation
y_forecasted = pred.predicted_mean
y_truth = df['1965-01-01':]
# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE of the forecasts is {}'.format(round(mse, 2)))

#=============================================
# get forecast 20 steps ahead in future
pred_uc = results.get_forecast(steps=20)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

# plotting forecasts ahead
ax = df.plot(label='Original data')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast values', title='Forecast plot with confidence interval')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
plt.legend()
plt.show()
#----------------------------------------------
