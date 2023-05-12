# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 22:36:14 2020

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot

series1 = read_excel('BuildingMaterialsWithMA.xls', sheet_name='MAdata', usecols=[1], header=0, squeeze=True)
series2 = read_excel('BuildingMaterialsWithMA.xls', sheet_name='MAdata', usecols=[2], header=0, squeeze=True)
series3 = read_excel('BuildingMaterialsWithMA.xls', sheet_name='MAdata', usecols=[3], header=0, squeeze=True)

series1.plot(label='Original series', color='black')
series2.plot(label='7MA', color='blue')
series3.plot(label='2x12MA', color='red')
pyplot.title('7MA and 2x12MA for building materials')
pyplot.legend()
pyplot.show()