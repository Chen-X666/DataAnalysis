# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:15:19 2020

@author: abz1e14
"""

# create an autocorrelation plot
from pandas import read_excel
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
series = read_excel('BuildingMaterials.xls', sheet_name='Data', header=0, index_col=0,
parse_dates=True, squeeze=True)
autocorrelation_plot(series)
pyplot.title('ACF plot building materials - line format')

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series, title='ACF plot building materials - histogram format', lags=100)
pyplot.show()

