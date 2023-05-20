# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:26:07 2021

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot
from numpy import sqrt
AustralianBeer  = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols = [1], 
                             header=0, squeeze=True, dtype=float)
NaiveF1  = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols = [2], 
                      header=0, squeeze=True, dtype=float)
NaiveF2 = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols=[3], 
                     header=0, squeeze=True, dtype=float)



# Evaluating the errors from both NF1 and NF2 methods
Error1 = AustralianBeer - NaiveF1
Error2 = AustralianBeer - NaiveF2
MSE1=sum(Error1**2)*1.0/len(NaiveF1)
MSE2=sum(Error2**2)*1.0/len(NaiveF2)

LowerForecast1 = NaiveF1 - 1.645*sqrt(MSE1)
UpperForecast1 = NaiveF1 + 1.645*sqrt(MSE1)

LowerForecast2 = NaiveF2 - 1.645*sqrt(MSE2)
UpperForecast2 = NaiveF2 + 1.645*sqrt(MSE2)

# Joint plot of original data and NF1 forecasts
AustralianBeer.plot(label='Original data')
NaiveF1.plot(label='NF1 forecast')
LowerForecast1.plot(label='NF1 lower bound')
UpperForecast1.plot(label='NF1 upper bound')
pyplot.legend()
pyplot.show()

# Joint plot of original data and NF2 forecasts
AustralianBeer.plot(label='Original data')
NaiveF2.plot(label='NF2 forecast')
LowerForecast2.plot(label='NF2 lower bound')
UpperForecast2.plot(label='NF2 upper bound')
pyplot.legend()
pyplot.show()