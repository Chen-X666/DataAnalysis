# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:23:01 2020

@author: abz1e14
"""
from pandas import read_excel
from matplotlib import pyplot
BuildingMaterials  = read_excel('BuildingMaterialsNF1NF2.xls', sheet_name='NF1NF2', usecols = [1], 
                             header=0, squeeze=True, dtype=float)
NaiveF1  = read_excel('BuildingMaterialsNF1NF2.xls', sheet_name='NF1NF2', usecols = [2], 
                      header=0, squeeze=True, dtype=float)
NaiveF2 = read_excel('BuildingMaterialsNF1NF2.xls', sheet_name='NF1NF2', usecols=[3], 
                     header=0, squeeze=True, dtype=float)

# Plot for the original data set
BuildingMaterials.plot(label='Original data', legend=True)
pyplot.show()

# Evaluating the errors from both NF1 and NF2 methods
Error1 = BuildingMaterials - NaiveF1
Error2 = BuildingMaterials - NaiveF2

# Plot of the error time series
Error1.plot(label='NF1 error plot', legend=True)
Error2.plot(label='NF2 error plot', legend=True)
pyplot.show()

# Creating an autocorrelation plot
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(Error1)
autocorrelation_plot(Error2)
pyplot.show()
