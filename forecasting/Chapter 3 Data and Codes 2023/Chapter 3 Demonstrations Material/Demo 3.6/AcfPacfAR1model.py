# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:19:41 2020

@author: abz1e14
"""

## removing seasonality from time series using seasonal differencing
from pandas import read_excel
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot

series = read_excel('DataAR1model.xls', sheet_name='ARdata', header=0, index_col=0, squeeze=True)

# Time, ACF, and PACF plots for original data
pyplot.plot(series)
pyplot.title('Time plot AR1 data')
plot_acf(series, title='ACF plot of AR1 data', lags=20)
plot_pacf(series, title='PACF plot of AR1 data', lags=20)
pyplot.show()