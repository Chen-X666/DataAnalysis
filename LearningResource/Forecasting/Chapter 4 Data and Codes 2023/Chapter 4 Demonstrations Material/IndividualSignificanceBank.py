# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:30:28 2020

@author: abz1e14
"""
from pandas import read_excel
from statsmodels.formula.api import ols
series = read_excel('Bank.xls', sheet_name='Data2', header=0, 
                     squeeze=True, dtype=float)

#reading the basic variables
DEOM = series.DEOM
AAA = series.AAA
Tto4 = series.Tto4
D3to4 = series.D3to4

#Regression model(s)
formula = 'DEOM ~ AAA + Tto4 + D3to4'

#Ordinary Least Squares (OLS)
results = ols(formula, data=series).fit()
print(results.summary())

# Here the main table is the second one,
# where all the statistics of the individual variables
# are given. 