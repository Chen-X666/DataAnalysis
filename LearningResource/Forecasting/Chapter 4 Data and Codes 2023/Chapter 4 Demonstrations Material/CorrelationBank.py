# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:30:28 2020

@author: abz1e14
"""
from pandas import read_excel
import matplotlib.pyplot as plt
import pandas as pd
series = read_excel('Bank.xls', sheet_name='Data3', header=0, 
                     squeeze=True, dtype=float)

#Plotting the scatter plots of each variable against the other one
pd.plotting.scatter_matrix(series, figsize=(8, 8))
plt.show()

# Correlation matrix for all the variables, 2 by 2
CorrelationMatrix = series.corr()
print(CorrelationMatrix)
# As in the case of Demo 1.4, corrcoef can be used for the variables in couples.