from pandas import read_excel
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

import dataAnalysisModel.statisticModel.ARIMA as ARIMA

plt.style.use('fivethirtyeight')

###############################
df = read_excel('PrintingWriting.xls', sheet_name='Data2', header=0,
               index_col=0, parse_dates=True, squeeze=True)
ARIMA.ts_plot(df)