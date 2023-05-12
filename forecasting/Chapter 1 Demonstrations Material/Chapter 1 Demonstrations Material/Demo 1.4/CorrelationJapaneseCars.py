# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:15:42 2020

@author: abz1e14
"""
from pandas import read_excel
import numpy as np
series1 = read_excel('JapaneseCars.xls', sheet_name='Data', usecols = [0], header=0, 
                     squeeze=True, dtype=float)
series2 = read_excel('JapaneseCars.xls', sheet_name='Data', usecols=[1], header=0, 
                     squeeze=True, dtype=float)
correlval=np.corrcoef(series1, series2) #generates result in a matrix format
#correlval=correlval[1,0] #to extract the most important value
print(correlval)