# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:05:53 2020

@author: abz1e14
"""
from pandas import read_excel
from statsmodels.tsa.api import Holt
from matplotlib import pyplot
series = read_excel('ShampooSales.xls', sheet_name='Data', header=0, 
              index_col=0, squeeze=True)

# Holt model 1: alpha = 0.8, beta=0.2
fit1 = Holt(series).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast1 = fit1.forecast(12).rename("Holt's linear trend")

fit2 = Holt(series, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast2 = fit2.forecast(12).rename("Exponential trend")

fit3 = Holt(series, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
fcast3 = fit3.forecast(12).rename("Additive damped trend")

fit4 = Holt(series).fit(optimized=True)
fcast4 = fit4.forecast(12).rename("Additive pooma damped trend")


fit1.fittedvalues.plot(color='blue')
fcast1.plot(color='blue', legend=True)

fit2.fittedvalues.plot(color='red')
fcast2.plot(color='red', legend=True)

fit3.fittedvalues.plot(color='green')
fcast3.plot(color='green', legend=True)

fcast4.plot(color='yellow', legend=True)

series.plot(color='black', legend=True)
pyplot.show()

#Evaluating the errors 
from sklearn.metrics import mean_squared_error 
MSE1=mean_squared_error(fit1.fittedvalues, series)
MSE2=mean_squared_error(fit2.fittedvalues, series)
MSE3=mean_squared_error(fit3.fittedvalues, series)

print('Summary of errors resulting from SES models 1, 2 & 3:')
import pandas as pd
cars = {'Model': ['MSE'],
        'LES model 1': [MSE1],
        'LES model 2': [MSE2],
        'LES model 3': [MSE3]
        }
AllErrors = pd.DataFrame(cars, columns = ['Model', 'LES model 1', 'LES model 2', 'LES model 3'])
print(AllErrors)
