# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:15:19 2020

@author: abz1e14
"""
from math import sqrt, log

# create an autocorrelation plot
import pandas as pd
import numpy as np
from pandas import read_excel
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.seasonal import seasonal_decompose

#The PACF of a non-stationary time series will typically have a large spike close to 1 at lag 1.
def Plot_PACF(series):
    #series = read_excel('DowJones.xls', sheet_name='Data2', header=0, index_col=0, parse_dates=True, squeeze=True)
    # PACF plot on 50 time lags
    plot_pacf(series, title='PACF of Dow Jones time series', lags=50)
    pyplot.show()

#The autocorrelations of stationary data drop to zero quite quickly
def Plot_ACF(series):
    # series = read_excel('Beer.xls', sheet_name='Data', header=0, index_col=0, parse_dates=True, squeeze=True)
    autocorrelation_plot(series) #from pandas - generate ACF in curve format
    plot_acf(series, title='ACF of Australian beer production data', lags=55) #from statsmodels - generates ACF in "lollipop plot" format
    pyplot.show()

#画自相关图
def Plot_Autocorrelation(series):
    autocorrelation_plot(series)
    pyplot.show()

def Plot_Seasonal_decompose(series):
    result = seasonal_decompose(series, model='additive')
    # result = seasonal_decompose(series, model='multiplicative')
    result.plot()
    pyplot.show()

def Plot_Log():
    series = read_excel('Electricity.xls',
                        sheet_name='Data', header=0,
                        index_col=0, parse_dates=True, squeeze=True)
    dataframe = pd.DataFrame(series.values)
    dataframe.columns = ['electricity']
    dataframe['electricity'] = log(dataframe['electricity'])
    pyplot.figure(1)
    # line plot
    pyplot.subplot(211)
    pyplot.plot(dataframe['electricity'])
    # histogram
    pyplot.subplot(212)
    pyplot.hist(dataframe['electricity'])
    pyplot.show()

def Plot_Sqrt():
    series = read_excel('Electricity.xls',
                        sheet_name='Data', header=0,
                        index_col=0, parse_dates=True, squeeze=True)
    dataframe = pd.DataFrame(series.values)
    dataframe.columns = ['electricity']
    dataframe['electricity'] = sqrt(dataframe['electricity'])
    pyplot.figure(1)
    # line plot
    pyplot.subplot(211)
    pyplot.plot(dataframe['electricity'])
    # histogram
    pyplot.subplot(212)
    pyplot.hist(dataframe['electricity'])
    pyplot.show()


