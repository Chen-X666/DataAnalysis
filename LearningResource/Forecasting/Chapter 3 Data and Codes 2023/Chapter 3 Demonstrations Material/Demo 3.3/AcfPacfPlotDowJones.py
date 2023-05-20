# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 06:24:23 2020

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
series = read_excel('DowJones.xls', sheet_name='Data2', header=0, index_col=0, parse_dates=True, squeeze=True)

# ACF plot on 50 time lags
plot_acf(series, title='ACF of Dow Jones time series', lags=50)

# PACF plot on 50 time lags
plot_pacf(series, title='PACF of Dow Jones time series', lags=50)
pyplot.show()