# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:34:10 2020

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = read_excel('HouseSales.xls', sheet_name='Data', header=0, 
              index_col=0, parse_dates=True, squeeze=True)
result = seasonal_decompose(series, model='additive')
#result = seasonal_decompose(series, model='multiplicative')
result.plot()
pyplot.show()
