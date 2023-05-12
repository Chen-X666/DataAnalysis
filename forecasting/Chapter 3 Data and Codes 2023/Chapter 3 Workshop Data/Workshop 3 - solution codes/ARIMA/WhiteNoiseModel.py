# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:42:21 2022

@author: abz1e14
"""
from random import gauss
from random import seed
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot

# seed random number generator
seed(1)
# create white noise series
series = [gauss(0.0, 1.0) for i in range(1000)]

# Once created, we can wrap the list in a Pandas Series for convenience.
series = Series(series)

# summary statistics of the artificially generated series
print('Statistics of the artificially generated series:')
print(series.describe())

# line plot of the artificially generated series
series.plot(title='Time plot of a white noise model')
pyplot.show()

# histogram plot of the artificially generated series
series.hist()

# ACF plot of an artificially generated white noise time series
plot_acf(series, title='ACF of a white noise model', lags=50)

# PACF plot of an artificially generated white noise time series
plot_pacf(series, title='PACF of a white noise model', lags=50)
pyplot.show()
