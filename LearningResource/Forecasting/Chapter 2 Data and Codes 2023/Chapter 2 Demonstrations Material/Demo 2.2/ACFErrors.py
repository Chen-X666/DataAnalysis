# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 16:18:20 2021

@author: abz1e14
"""

from pandas import read_excel
from matplotlib import pyplot
AustralianBeer  = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols = [1], 
                             header=0, squeeze=True, dtype=float)
NaiveF1  = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols = [2], 
                      header=0, squeeze=True, dtype=float)
NaiveF2 = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols=[3], 
                     header=0, squeeze=True, dtype=float)


# Evaluating the errors from both NF1 and NF2 methods
Error1 = AustralianBeer - NaiveF1
Error2 = AustralianBeer - NaiveF2

# ACF plots for original data, Error1 & Error2
#===============================
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(AustralianBeer, lags=50)
plot_acf(Error1, lags=50)
plot_acf(Error2, lags=50)
pyplot.show()