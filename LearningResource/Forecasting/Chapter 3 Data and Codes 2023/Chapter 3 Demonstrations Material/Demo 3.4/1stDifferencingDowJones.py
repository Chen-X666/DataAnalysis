# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:23:06 2020

@author: abz1e14
"""

## detrend a time series using differencing
from pandas import read_excel
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot

series = read_excel('DowJones.xls', sheet_name='Data2', header=0, index_col=0, parse_dates=True, squeeze=True)
X = series.values
diff = list()
for i in range(1, len(X)):
	value = X[i] - X[i - 1]
	diff.append(value)
pyplot.plot(diff)
pyplot.title('Time plot Dow Jones 1st difference')

# ACF plot of time series
plot_acf(diff, title='ACF of Dow Jones 1st difference', lags=50)

# PACF plot of time series
plot_pacf(diff, title='PACF of Dow Jones 1st difference', lags=50)
pyplot.show()