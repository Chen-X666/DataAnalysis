# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:05:53 2020

@author: abz1e14
"""
from pandas import read_excel
import pandas as pd
from statsmodels.tsa.api import Holt
from matplotlib import pyplot
series = read_excel('EmploymentPrivateServices.xls', header=0, 
              index_col=0, parse_dates=True, squeeze=True)

# ===================================
# Simple Exponential Smoothing #
# ===================================
# Holt model 1: alpha = 0.5, beta=0.5
fit1 = Holt(series).fit(smoothing_level=0.5, smoothing_trend=0.5, optimized=False)
fcast1 = fit1.forecast(12).rename("Model 1: Holt's linear trend")
#------------------------------------
fit2 = Holt(series, exponential=True).fit(smoothing_level=0.2, smoothing_trend=0.4, optimized=False)
fcast2 = fit2.forecast(12).rename("Model 2: Exponential trend")
#------------------------------------
fit3 = Holt(series, damped=True).fit()
fcast3 = fit3.forecast(12).rename("Model 3: Damped trend + optimized")
#------------------------------------
fit4 = Holt(series).fit(optimized=True)
fcast4 = fit4.forecast(12).rename("Model 4: Linear trend + optimized")

#=====================================
# Time and forecast plots 
#=====================================
series.plot(color='black', legend=True)
#-------------------------------------
fit1.fittedvalues.plot(color='blue')
fcast1.plot(color='blue', legend=True)
#-------------------------------------
fit2.fittedvalues.plot(color='red')
fcast2.plot(color='red', legend=True)
#-------------------------------------
fit3.fittedvalues.plot(color='green')
fcast3.plot(color='green', legend=True)
#-------------------------------------
fit4.fittedvalues.plot(color='yellow')
fcast4.plot(color='yellow', legend=True)
#-------------------------------------
pyplot.xlabel('Dates')
pyplot.ylabel('Values')
pyplot.title("Holt's method-based forecasts for Employment Services")
pyplot.show()

#====================================
# Evaluating the errors 
#====================================
from sklearn.metrics import mean_squared_error 
MSE1=mean_squared_error(fit1.fittedvalues, series)
MSE2=mean_squared_error(fit2.fittedvalues, series)
MSE3=mean_squared_error(fit3.fittedvalues, series)
MSE4=mean_squared_error(fit4.fittedvalues, series)

#=====================================
# Printing the paramters and errors of the methods
#=====================================
print('Summary of paramters and errors from each of the models:')
params = ['smoothing_level', 'smoothing_trend', 'damping_trend', 'initial_level', 'initial_trend']
results=pd.DataFrame(index=[r"alpha", r"beta", r"phi", r"l0", r"b0", "MSE"] ,columns=["Holt model 1", "Holt model 2","Holt model 3", "Holt model 4"])
results["Holt model 1"] =            [fit1.params[p] for p in params] + [MSE1]
results["Holt model 2"] =         [fit2.params[p] for p in params] + [MSE2]
results["Holt model 3"] =    [fit3.params[p] for p in params] + [MSE3]
results["Holt model 4"] =       [fit4.params[p] for p in params] + [MSE4]
print(results)

print(fit1.fittedvalues)
print(fcast1)