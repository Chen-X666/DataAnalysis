# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:21:21 2020

@author: abz1e14
"""
from pandas import read_excel
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from matplotlib import pyplot
series = read_excel('CementProduction.xls', sheet_name='Data', header=0, 
              index_col=0, parse_dates=True, squeeze=True)

# ===================================
# Holt-Winter method in different scenarios # 
# ===================================
# ===================================
# Model 1: Holt-Winter method with additive trend and seasonality 
# Here, alpha = 0.3, beta=0.5, gamma=0.7
# ===================================
fit1 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add').fit(smoothing_level = 0.3, smoothing_trend=0.5,  smoothing_seasonal=0.7)
fit1.fittedvalues.plot(color='red')

# ===================================
# Model 2: Holt-Winter method with additive trend and multiplicative seasonality 
# Here, alpha = 0.3, beta=0.5, gamma=0.7
# ===================================
fit2 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='mul').fit(smoothing_level = 0.3, smoothing_trend=0.5,  smoothing_seasonal=0.7)
fit2.fittedvalues.plot(color='blue')

# ===================================
# Model 3: Holt-Winter method with additive trend and seasonality 
# Here, the parameters alpha, beta, and gamma are optimized
# ===================================
fit3 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add').fit()
fit3.fittedvalues.plot(color='green')

# ===================================
# Model 4: Holt-Winter method with additive trend and multiplicative seasonality 
# Here, the parameters alpha, beta, and gamma are optimized
# ===================================
fit4 = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='mul').fit()
fit4.fittedvalues.plot(color='yellow')

print("Forecasting Cement Production with Holt-Winters method")
#=====================================
# Time and forecast plots 
#=====================================
series.rename('Time plot of original series').plot(color='black', legend=True)
fit1.forecast(12).rename('Model 1: HW-additive seasonality').plot(color='red', legend=True)
fit2.forecast(12).rename('Model 2: HW-multiplicative seasonality').plot(color='blue', legend=True)
fit3.forecast(12).rename('Model 3: Opt HW-additive seasonality').plot(color='green', legend=True)
fit4.forecast(12).rename('Model 4: Opt HW-multiplicative seasonality').plot(color='yellow', legend=True)
pyplot.xlabel('Dates')
pyplot.ylabel('Values')
pyplot.title('HW method-based forecasts for cement production')
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
# Printing the paramters and errors for each scenario
#=====================================
results=pd.DataFrame(index=[r"alpha", r"beta", r"gamma", r"l0", "b0", "MSE"])
params = ['smoothing_level', 'smoothing_trend', 'smoothing_seasonal', 'initial_level', 'initial_trend']
results["HW model 1"] = [fit1.params[p] for p in params] + [MSE1]
results["HW model 2"] = [fit2.params[p] for p in params] + [MSE2]
results["HW model 3"] = [fit3.params[p] for p in params] + [MSE3]
results["HW model 4"] = [fit4.params[p] for p in params] + [MSE4]
print(results)

#=====================================
# Evaluating and plotting the residual series for each scenario
#=====================================
residuals1= fit1.fittedvalues - series
residuals2= fit2.fittedvalues - series
residuals3= fit3.fittedvalues - series
residuals4= fit4.fittedvalues - series
residuals1.rename('residual plot for model 1').plot(color='red', legend=True)
residuals2.rename('residual plot for model 2').plot(color='blue', legend=True)
residuals3.rename('residual plot for model 3').plot(color='green', legend=True)
residuals4.rename('residual plot for model 4').plot(color='yellow', legend=True)
pyplot.title('Residual plots for models 1-4')
pyplot.show()

#=====================================
# ACF plots of the residual series for each scenario
#=====================================
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals1, title='Residual ACF for model 1', lags=50)
plot_acf(residuals2, title='Residual ACF for model 2', lags=50)
plot_acf(residuals3, title='Residual ACF for model 3', lags=50)
plot_acf(residuals4, title='Residual ACF for model 4', lags=50)
pyplot.show()



