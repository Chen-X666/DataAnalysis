# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 21:18:04 2020

@author: abz1e14
"""
##Change data file and remove unnecessary one
#from pandas import read_excel
#from matplotlib import pyplot
#series = read_excel('Electricity.xls',
#              sheetname='SeasonalData', header=0, index_col=0, parse_dates=True, squeeze=True)  # you can include various other parameters
#series.plot()
#pyplot.show()


from pandas import read_excel
import matplotlib.pyplot as plt
import numpy as np
series = read_excel('Electricity.xls',
              sheet_name='SeasonalData', header=0, index_col=0, parse_dates=True, squeeze=True)
x = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
months = ['Jan','Feb','Mar','Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.xticks(x, months)
plt.plot(x, series)
plt.show()