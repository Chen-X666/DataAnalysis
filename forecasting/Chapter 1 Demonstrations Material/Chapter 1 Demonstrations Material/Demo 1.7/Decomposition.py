# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 08:32:41 2021

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
#series = read_excel('CementProduction.xls', sheetname='Data', header=0, 
#              index_col=0, parse_dates=True, squeeze=True)
#result = seasonal_decompose(series, model='additive')
series = read_excel('ClayBricks.xls', sheet_name='BRICKSQ', header=0, 
              index_col=0, parse_dates=True, squeeze=True)
#series = read_excel('Electricity.xls',
#              sheetname='ELEC', header=0, 
#              index_col=0, parse_dates=True, squeeze=True)
#
#series = read_excel('TreasuryBills.xls',
#              sheetname='USTREAS', header=0, 
#              index_col=0, parse_dates=True, squeeze=True) 

#series = read_excel('DowJones.xls', sheet_name='Data2', header=0, index_col=0, parse_dates=True, squeeze=True)

result = seasonal_decompose(series, model='additive')
result.plot()
pyplot.show()


# the following optional commands can be used to extract the values of the decomposition components
residual = result.resid
seasonal = result.seasonal 
trend = result.trend
