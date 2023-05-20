# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:47:14 2022

@author: abz1e14
"""
from pandas import read_excel
import matplotlib.pyplot as plt
series = read_excel('BuildingMaterials.xls', sheetname='Data', header=0,
              index_col=0, parse_dates=True, squeeze=True)
series.plot(color='red')
plt.xlabel('Dates')
plt.ylabel('Production values')
plt.title('Building materials production from 1986 to 2008')
plt.show()
#------------------------
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
series = read_excel('BuildingMaterials.xls', sheet_name='Data', usecols = [1],
                             header=0, squeeze=True, dtype=float)
# ACF plot on 50 time lags
plot_acf(series, title='ACF of building materials time series', lags=50)

# PACF plot on 50 time lags
plot_pacf(series, title='PACF of building materials time series', lags=50)
pyplot.show()
#--------------------------------------------------
# calculate stationarity test of time series data
#from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
#series = read_csv('airline-passengers.csv', header=0, index_col=0, parse_dates=True,squeeze=True)
X = series.values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
