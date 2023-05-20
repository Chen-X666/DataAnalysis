# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:36:14 2020

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot
series = read_excel('MilkProduction.xls',
              sheet_name='AdjustedData', header=0, 
              index_col=0, parse_dates=True, squeeze=True)  # you can include various other parameters
series.plot()
pyplot.show()
