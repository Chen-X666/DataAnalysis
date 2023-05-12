# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:15:42 2020

@author: abz1e14
"""
from pandas import read_excel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
series1 = read_excel('BuildingMaterials.xls', sheet_name='CorelData', usecols = [1], header=0, 
                     squeeze=True, dtype=float)
series2 = read_excel('BuildingMaterials.xls', sheet_name='CorelData', usecols=[2], header=0, 
                     squeeze=True, dtype=float)
correlval=np.corrcoef(series1, series2)
correlval=correlval[1,0]
print('The correlation between building materials and cement production is:', correlval)

BuildingCement = {'Building': series1, 'Cement': series2}
df = pd.DataFrame(BuildingCement, columns=['Building', 'Cement'])
plt.scatter(df.Building, df.Cement)

plt.xlabel('Building Materials')
plt.ylabel('Cement Production')
plt.title('Cement Production/Building Materials Relationship in Australia')