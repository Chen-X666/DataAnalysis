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

#reading the indicator variables
D1=series.D1
D2=series.D2
D3=series.D3
D4=series.D4
D5=series.D5
D6=series.D6
D7=series.D7
D8=series.D8
D9=series.D9
D10=series.D10
D11=series.D11

#reading the time variable
time=series.time

#Regression model(s)
formula1 = 'DEOM ~ AAA + Tto4 + D3to4'
formula2 = 'DEOM ~ AAA + Tto4 + D3to4 + D1 + D2 + D3+D4+D5+D6+D7+D8+D9+D10+D11'
formula3 = 'DEOM ~ AAA + Tto4 + D3to4 + D1 + D2 + D3+D4+D5+D6+D7+D8+D9+D10+D11+time'

#Ordinary Least Squares (OLS)
results1 = ols(formula1, data=series).fit()
results2 = ols(formula2, data=series).fit()
results3 = ols(formula3, data=series).fit()
print(results1.summary())
print(results2.summary())
print(results3.summary())

# the results from IndividualSignificance.py, 
# StatsWithIndicatorsBank.py, 
# and StatsWithIndicatorsTimeBank.py are summarised 
# for easy comparison of the key statistics 