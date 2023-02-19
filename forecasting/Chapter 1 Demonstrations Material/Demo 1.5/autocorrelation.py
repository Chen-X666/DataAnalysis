# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 17:37:31 2021

@author: abz1e14
"""

#generates seasonal and ACF plots to demonstrate seasonality

from pandas import read_excel
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot 
from statsmodels.graphics.tsaplots import plot_acf
series1 = read_excel('CementProduction.xls', sheet_name='Data', header=0, 
              index_col=0, parse_dates=True)
series2 = read_excel('CementProduction.xls', sheet_name='SeasonalData', header=0,
                    index_col=0, parse_dates=True, squeeze=True)
series2.plot(title='Seasonal plots building materials time series')
pyplot.show()
print(series1)

plot_acf(series1, title='ACF plot of building materials time series', lags=60)
pyplot.show()


autocorrelation_plot(series1)
pyplot.show()