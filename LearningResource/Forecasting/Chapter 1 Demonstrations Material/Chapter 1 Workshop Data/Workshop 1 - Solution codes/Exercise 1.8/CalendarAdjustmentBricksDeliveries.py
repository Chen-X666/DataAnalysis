# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:36:14 2020

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot
from pandas import DataFrame
from numpy import log
from numpy import sqrt
series = read_excel('BricksDeliveries.xls', sheet_name='CalAdjData', header=0, squeeze=True)

OriginalData=series.Yt
AdjustedData=series.Adjusted 
OriginalData.plot(label='Original series')
AdjustedData.plot(label='Adjusted series')
pyplot.title('Calendar adjustment for bricks deliveries data')
pyplot.legend()
pyplot.show()


#-----------------------
# Log transform
#-----------------------
dataframe1 = DataFrame(AdjustedData.values)
dataframe1.columns = ['bricks']
dataframe1['bricks'] = log(dataframe1['bricks'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe1['bricks'])
pyplot.title('Log transform-calendar adjustment brick deliveries')
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe1['bricks'])
pyplot.show()


#-----------------------
# Sqrt transform
#-----------------------
dataframe2 = DataFrame(AdjustedData.values)
dataframe2.columns = ['bricks']
dataframe2['bricks'] = sqrt(dataframe2['bricks'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe2['bricks'])
pyplot.title('Sqrt transform-calendar adjustment brick deliveries')
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe2['bricks'])
pyplot.show()