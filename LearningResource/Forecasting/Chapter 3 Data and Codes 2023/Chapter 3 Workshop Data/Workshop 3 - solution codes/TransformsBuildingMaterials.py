# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:47:30 2020

@author: abz1e14
"""

from pandas import read_excel
from pandas import DataFrame
from numpy import sqrt, log
from matplotlib import pyplot
series = read_excel('BuildingMaterials.xls', sheet_name='Data', usecols = [1], 
                             header=0, squeeze=True, dtype=float) 
dataframe = DataFrame(series.values)
dataframe.columns = ['Building']

#================
# Sqrt transform
#================
dataframe['Building'] = sqrt(dataframe['Building'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['Building'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Building'])
pyplot.show()

#================
# Log transform
#================
dataframe['Building'] = log(dataframe['Building'])
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['Building'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['Building'])
pyplot.show()