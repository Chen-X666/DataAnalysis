# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 09:20:50 2021

@author: abz1e14
"""

from pandas import read_excel
import matplotlib.pyplot as plt
import numpy as np
series = read_excel('ClayBricks.xls',
              sheet_name='SeasonalData', header=0, index_col=0, parse_dates=True, squeeze=True)
x = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
months = ['Jan','Feb','Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(x, months)
plt.plot(x, series)
plt.show()