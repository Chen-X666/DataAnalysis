# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 22:23:06 2020

@author: abz1e14
"""

## detrend a time series using differencing
from pandas import read_excel
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot

series = read_excel('BuildingMaterials.xls', sheet_name='Data', usecols = [1], 
                             header=0, squeeze=True, dtype=float)
X = series.values
print(len(series))
diff = list()
for i in range(1, len(X)):
	value = X[i] - X[i - 12]
	diff.append(value)
print(len(diff))
# pyplot.plot(diff)
# pyplot.title('Time plot building materials 1st difference')
#
# # ACF plot of time series
# plot_acf(diff, title='ACF of building materials 1st difference', lags=50)
#
# # PACF plot of time series
# plot_pacf(diff, title='PACF of building materials 1st difference', lags=50)
# pyplot.show()