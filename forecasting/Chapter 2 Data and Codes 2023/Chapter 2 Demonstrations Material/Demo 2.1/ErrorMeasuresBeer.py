# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:18:48 2020

@author: abz1e14
"""
from pandas import read_excel
from matplotlib import pyplot
AustralianBeer  = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols = [1], 
                             header=0, squeeze=True, dtype=float)
NaiveF1  = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols = [2], 
                      header=0, squeeze=True, dtype=float)
NaiveF2 = read_excel('BeerErrorsData.xls', sheet_name='NF1NF2', usecols=[3], 
                     header=0, squeeze=True, dtype=float)


# Joint plot of original data and NF1 forecasts
AustralianBeer.plot(legend=True)
NaiveF1.plot(legend=True)
pyplot.show()

# Joint plot of original data and NF2 forecasts
AustralianBeer.plot(legend=True)
NaiveF2.plot(legend=True)
pyplot.show()

# Evaluating the errors from both NF1 and NF2 methods
Error1 = AustralianBeer - NaiveF1
Error2 = AustralianBeer - NaiveF2
ME1 = sum(Error1)* 1.0/len(NaiveF1)
ME2 = sum(Error2)* 1.0/len(NaiveF2)
MAE1=sum(abs(Error1))*1.0/len(NaiveF1)
MAE2=sum(abs(Error2))*1.0/len(NaiveF2)
MSE1=sum(Error1**2)*1.0/len(NaiveF1)
MSE2=sum(Error2**2)*1.0/len(NaiveF2)

PercentageError1=(Error1/AustralianBeer)*100
PercentageError2=(Error2/AustralianBeer)*100
MPE1 = sum(PercentageError1)* 1.0/len(NaiveF1)
MPE2 = sum(PercentageError2)* 1.0/len(NaiveF2)
MAE1=sum(abs(PercentageError1))*1.0/len(NaiveF1)
MAE2=sum(abs(PercentageError2))*1.0/len(NaiveF2)


#Printing a summary of the errors in a tabular form
print('Summary of errors resulting from NF1 & NF2:')
import pandas as pd
cars = {'Errors': ['ME','MAE','MSE','MPE', 'MAPE'],
        'NF1': [ME1, MAE1, MSE1, MPE1, MAE1],
        'NF2': [ME2, MAE2, MSE2, MPE2, MAE2]
        }
AllErrors = pd.DataFrame(cars, columns = ['Errors', 'NF1', 'NF2'])
print(AllErrors)