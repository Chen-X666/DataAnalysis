# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:37:50 2020

@author: abz1e14
"""
from pandas import read_excel
from matplotlib import pyplot
series = read_excel('Beer.xls',
              sheet_name='Data', header=0, 
              index_col=0, parse_dates=True, squeeze=True)  # you can include various other parameters
series.plot()
pyplot.show()
