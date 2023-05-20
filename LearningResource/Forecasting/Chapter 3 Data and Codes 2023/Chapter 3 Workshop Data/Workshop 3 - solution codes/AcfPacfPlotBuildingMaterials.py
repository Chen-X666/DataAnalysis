# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 06:24:23 2020

@author: abz1e14
"""

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