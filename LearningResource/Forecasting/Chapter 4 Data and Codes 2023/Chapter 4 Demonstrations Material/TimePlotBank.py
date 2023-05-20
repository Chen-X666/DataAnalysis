# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:30:28 2020

@author: abz1e14
"""
from pandas import read_excel
import matplotlib.pyplot as plt
series = read_excel('Bank.xls', sheet_name='Data2', header=0, 
                     squeeze=True, dtype=float)

#reading the basic variables
DEOM = series.DEOM
AAA = series.AAA
Tto4 = series.Tto4
D3to4 = series.D3to4

DEOM.plot()
plt.xlabel('time')
plt.ylabel('Difference end of month balance')
plt.title('DEOM')
plt.show()

AAA.plot()
plt.xlabel('time')
plt.ylabel('Composite AAA Bond rates')
plt.title('AAA')
plt.show()

Tto4.plot()
plt.xlabel('time')
plt.ylabel('US Govt 3-4 year Bond rates')
plt.title('3to4')
plt.show()

D3to4.plot()
plt.xlabel('time')
plt.ylabel('Difference US Govt 3-4 year Bond rates')
plt.title('D3to4')
plt.show()
