# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:18:04 2020

@author: abz1e14
"""
#Change data file and remove unnecessary one
from pandas import read_excel
from matplotlib import pyplot
series = read_excel('Beer.xls', sheet_name='SeasData', header=0,
                    index_col=0, parse_dates=True, squeeze=True)  # you can include various other parameters
series.plot()
pyplot.show()

