# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:47:30 2020

@author: abz1e14
"""

from pandas import read_excel
from pandas import DataFrame
from numpy import log
from numpy import sqrt
from matplotlib import pyplot
series = read_excel('BricksDeliveries.xls', header=0, 
              index_col=0, parse_dates=True, squeeze=True) 

## Time plot of original time series
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(series)
pyplot.title('Time plot for brick deliveries data')
# histogram
pyplot.subplot(212)
pyplot.hist(series)
pyplot.show()

#-----------------------
# Log transform
#-----------------------
dataframe1 = DataFrame(series.values)
dataframe1.columns = ['bricks']
dataframe1['bricks'] = log(dataframe1['bricks'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe1['bricks'])
pyplot.title('Log transform for brick deliveries data')
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe1['bricks'])
pyplot.show()

#-----------------------
# Sqrt transform
#-----------------------
dataframe2 = DataFrame(series.values)
dataframe2.columns = ['bricks']
dataframe2['bricks'] = sqrt(dataframe2['bricks'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe2['bricks'])
pyplot.title('Sqrt transform for brick deliveries data')
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe2['bricks'])
pyplot.show()