# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:18:04 2020

@author: abz1e14
"""
from pandas import read_excel
import matplotlib.pyplot as plt
import numpy as np

series = read_excel('BuildingMaterials.xls',
              sheet_name='SeasonalData', header=0, 
              index_col=0, squeeze=True)
x = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
months = ['Jan','Feb','Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(x, months)
plt.plot(x, series)
plt.show()