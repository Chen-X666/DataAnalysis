# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:53:16 2022

@author: abz1e14
"""

from pandas import read_excel
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

series = read_excel('BuildingMaterials.xls', sheet_name='Data', header=0,
                index_col=0, parse_dates=True, squeeze=True)

#ARIMA(1,1,2) model
model = ARIMA(series, order=(1,1,2))

# generates ARIMA Model Results table
model_fit = model.fit()
fitted = model.fit()
print(model_fit.summary())

# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()

# Forecast
fc, se, conf = fitted.forecast(15, alpha=0.005)  # 95% conf

pred_ci = pred_uc.conf_int()
# plotting forecasts ahead
ax = series.plot(label='Original data')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast values', title='Forecast plot with confidence interval')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
plt.legend()
plt.show()
#====================================================================

#====================================================================
# MSE evaluation
y_forecasted = pred.predicted_mean
y_truth = series['2000-01-01':]
# Compute the mean square error
MSE = ((y_forecasted - y_truth) ** 2).mean()
print('MSE of the forecasts is {}'.format(round(MSE, 2)))