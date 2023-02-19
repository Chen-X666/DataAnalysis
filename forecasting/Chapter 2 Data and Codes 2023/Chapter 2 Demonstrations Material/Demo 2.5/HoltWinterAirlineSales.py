# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:21:21 2020

@author: abz1e14
"""
from pandas import read_excel
from statsmodels.tsa.api import ExponentialSmoothing
from matplotlib import pyplot
series = read_excel('AirlineSales.xls', sheet_name='Data', header=0, 
              index_col=0, parse_dates=True, squeeze=True)

## Model 1: alpha = 0.3, beta=0.5, gamma=0.7
fit1 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add').fit(smoothing_level = 0.3, smoothing_slope=0.5,  smoothing_seasonal=0.7)
fit1.fittedvalues.plot(color='red')
fit1.forecast(12).plot(color='red', legend=True)

fit2 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='mul').fit()
fit2.fittedvalues.plot(color='blue')
fit2.forecast(12).plot(color='blue', legend=True)

#fit3 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
#fit3.fittedvalues.plot(color='green')
#fit3.forecast(12).plot(color='green', legend=True)
#
#fit4 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)
#fit4.fittedvalues.plot(color='yellow')
#fit4.forecast(12).plot(color='yellow', legend=True)


print("Forecasting Airline Sales with Holt-Winters: Additive + Multiplicative")

series.plot(color='black', legend=True)
pyplot.show()
