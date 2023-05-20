# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 23:00:14 2022

@author: abz1e14
"""
#===================================================
#Code for identifying the parameters with smallest AIC
#===================================================
from pandas import read_excel
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
import itertools
plt.style.use('fivethirtyeight')

series = read_excel('BuildingMaterials.xls', sheet_name='Data', header=0, index_col=0, parse_dates=True, squeeze=True)

#Define the p, d and q parameters to take any value between 0 and 1
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets (i.e., P, D, Q)
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

# Indentification of best model from different combinations of pdq and seasonal_pdq
warnings.filterwarnings("ignore") # specify to ignore warning messages
best_score, best_param, best_paramSeasonal = float("inf"), None, None
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(series, order=param, seasonal_order=param_seasonal, enforce_invertibility=False)
            results = mod.fit()
            if results.aic < best_score:
                best_score, best_param, best_paramSeasonal = results.aic, param, param_seasonal
            print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
print('The best model is ARIMA{}x{} - AIC:{}'.format(best_param, best_paramSeasonal, best_score))
