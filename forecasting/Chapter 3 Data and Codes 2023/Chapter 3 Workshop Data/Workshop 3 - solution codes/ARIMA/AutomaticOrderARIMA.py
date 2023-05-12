# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:51:59 2022

@author: abz1e14
"""
from matplotlib import pyplot as plt
#===================================================
#Code for identifying the parameters with smallest AIC
#===================================================
from pandas import read_excel
from statsmodels.tsa.arima.model import ARIMA
import warnings
import itertools
import statsmodels.api as sm

series = read_excel('BuildingMaterials.xls', sheet_name='Data', header=0, index_col=0, parse_dates=True, squeeze=True)

#define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 6)

#generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

#indentification of best model from different combinations of pdq
warnings.filterwarnings("ignore") # specify to ignore warning messages
best_score, best_param = float("inf"), None
for param in pdq:
            print(param)
        # try:
            mod = ARIMA(series, order=param)
            results = mod.fit()
            if results.aic < best_score:
                best_score, best_param = results.aic, param
            print('ARIMA{} - AIC:{}'.format(param, results.aic))
        # except:
        #     continue
print('The best model is ARIMA{} - AIC:{}'.format(best_param, best_score))

# Fit ARIMA model with best parameters
model = sm.tsa.arima.ARIMA(series, order=best_param)
results = model.fit()

# Print summary of model results
print(results.summary())

# Generate predictions for next 12 time steps
predictions = results.forecast(20)
print(predictions)

# # Calculate MSE and AIC
# mse = ((predictions - test_data.values.squeeze()) ** 2).mean()
aic = results.aic

# print(f"Mean squared error (MSE): {mse:.4f}")
print(f"Akaike Information Criterion (AIC): {aic:.4f}")

# Plot the actual and predicted values
plt.plot(series.index, series.values, label='Original Data')
# plt.plot(test_data.index, test_data.values, label='Test Data')
plt.plot(predictions.index, predictions.values, label='Predictions')
plt.legend()
plt.show()

results.plot_diagnostics(figsize=(6, 6))
plt.show()

