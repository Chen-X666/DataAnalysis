# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:56:03 2020

@author: abz1e14
"""
from pandas import read_excel
from statsmodels.tsa.api import SimpleExpSmoothing
from matplotlib import pyplot
series = read_excel('ShampooSales.xls', sheet_name='Data', header=0, 
              index_col=0, parse_dates=True, squeeze=True)
print(series)
# Simple Exponential Smoothing #

## SES model 1: alpha = 0.1
fit1 = SimpleExpSmoothing(series).fit(smoothing_level=0.1,optimized=False)
fcast1 = fit1.forecast(100).rename(r'$\alpha=0.1$')
# Plot of fitted values and forecast of next 10 values, respectively
fit1.fittedvalues.plot(color='blue')
fcast1.plot(color='blue', legend=True)

## SES model 2: alpha = 0.7
fit2 = SimpleExpSmoothing(series).fit(smoothing_level=0.7,optimized=False)
fcast2 = fit2.forecast(100).rename(r'$\alpha=0.7$')
# Plot of fitted values and forecast of next 10 values, respectively
fcast2.plot(color='red', legend=True)
fit2.fittedvalues.plot(color='red')

## SES model 3: alpha automatically selected by the built-in optimization software
fit3 = SimpleExpSmoothing(series).fit()
fcast3 = fit3.forecast(100).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])
# Plot of fitted values and forecast of next 10 values, respectively
fcast3.plot(color='green', legend=True)
fit3.fittedvalues.plot(color='green')

# Plotting the original data together with the 3 forecast plots
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
        'SES model 1': [MSE1],
        'SES model 2': [MSE2],
        'SES model 3': [MSE3]
        }
AllErrors = pd.DataFrame(cars, columns = ['Model', 'SES model 1', 'SES model 2', 'SES model 3'])
print(AllErrors)
