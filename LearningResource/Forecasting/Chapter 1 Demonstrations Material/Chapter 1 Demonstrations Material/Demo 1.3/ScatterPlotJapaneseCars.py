# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 18:21:22 2020

@author: abz1e14
"""

import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np
from pandas import read_excel
series1 = read_excel('JapaneseCars.xls', sheet_name='Data', usecols = [0], header=0, 
                     squeeze=True, dtype=float)
series2 = read_excel('JapaneseCars.xls', sheet_name='Data', usecols=[1], header=0, 
                     squeeze=True, dtype=float)

Japanese = {'Mileage': series1, 'Price': series2}
df = pd.DataFrame(Japanese, columns=['Mileage', 'Price'])
plt.scatter(df.Mileage, df.Price)

plt.xlabel('Mileage (MPG)')
plt.ylabel('Price (US$)')
plt.title('Price/Mileage Relationship for 19 Japanese cars')
