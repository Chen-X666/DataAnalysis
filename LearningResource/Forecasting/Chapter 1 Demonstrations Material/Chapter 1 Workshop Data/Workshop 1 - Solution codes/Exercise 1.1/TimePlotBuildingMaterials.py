# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 19:37:50 2020

@author: abz1e14
"""
from pandas import read_excel
import matplotlib.pyplot as plt
series = read_excel('BuildingMaterials.xls', sheet_name='Data', header=0, 
              index_col=0, parse_dates=True, squeeze=True)  # you can include various other parameters
series.plot(color='red')
plt.xlabel('Dates')
plt.ylabel('Production values')
plt.title('Building materials production from 1986 to 2008')
plt.show()

