# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:50:51 2022

@author: abz1e14
"""

from pandas import read_excel
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib import pyplot

series = read_excel('DataMA1model.xls', sheet_name='MAdata', usecols = [1],
                    header=0, squeeze=True)

# Time, ACF, and PACF plots for original data
pyplot.plot(series)
pyplot.title('Time plot MA1 data')
plot_acf(series, title='ACF plot of MA1 data', lags=20)
plot_pacf(series, title='PACF plot of MA1 data', lags=20)
pyplot.show()