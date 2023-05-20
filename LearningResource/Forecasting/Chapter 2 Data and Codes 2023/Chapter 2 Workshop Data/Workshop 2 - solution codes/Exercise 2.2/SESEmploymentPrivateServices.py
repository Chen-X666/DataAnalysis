# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 10:56:03 2020

@author: abz1e14
"""
from pandas import read_excel
from statsmodels.tsa.api import SimpleExpSmoothing
from matplotlib import pyplot
series = read_excel('EmploymentPrivateServices.xls', header=0, 
              index_col=0, parse_dates=True, squeeze=True)

# ==============================
# Simple Exponential Smoothing #
# ==============================
## SES model 1: alpha = 0.5
fit1 = SimpleExpSmoothing(series).fit(smoothing_level=0.5, optimized=False)
fcast1 = fit1.forecast(10).rename(r'$\alpha=0.5$')
# Plot of fitted values and forecast of next 10 values, respectively
fit1.fittedvalues.plot(color='blue')
fcast1.plot(color='blue', legend=True)

## SES model 2: alpha = 0.8
fit2 = SimpleExpSmoothing(series).fit(smoothing_level=0.8, optimized=False)
fcast2 = fit2.forecast(10).rename(r'$\alpha=0.8$')
# Plot of fitted values and forecast of next 10 values, respectively
fcast2.plot(color='red', legend=True)
fit2.fittedvalues.plot(color='red')

## SES model 3: alpha automatically selected by the built-in optimization software
fit3 = SimpleExpSmoothing(series).fit(optimized=True)
fcast3 = fit3.forecast(10).rename(r'$\alpha=%s$'%fit3.model.params['smoothing_level'])
# Plot of fitted values and forecast of next 10 values, respectively
fcast3.plot(color='green', legend=True)
fit3.fittedvalues.plot(color='green')

# Plotting the original data together with the 3 forecast plots
series.plot(color='black', legend=True)
pyplot.xlabel('Dates')
pyplot.ylabel('Values')
pyplot.title('SES method-based forecasts for Employment Services')
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
