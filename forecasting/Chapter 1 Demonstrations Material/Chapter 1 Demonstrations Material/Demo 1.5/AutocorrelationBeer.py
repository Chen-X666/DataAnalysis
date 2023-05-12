# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:15:19 2020

@author: abz1e14
"""

# create an autocorrelation plot
from pandas import read_excel
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot 
from statsmodels.graphics.tsaplots import plot_acf
series = read_excel('Beer.xls', sheet_name='Data', header=0, index_col=0, parse_dates=True, squeeze=True)
autocorrelation_plot(series) #from pandas - generate ACF in curve format
plot_acf(series, title='ACF of Australian beer production data', lags=55) #from statsmodels - generates ACF in "lollipop plot" format 
pyplot.show()


