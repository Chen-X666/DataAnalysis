# -*- coding: utf-8 -*-
"""
判断时间序列是否为平稳序列的方式有两种，一是通过单位根检验（如DF、ADF、PP方法等）； 二是通过观察时间序列的自相关（Autocorrelation Coefficient，ACF）和
偏自相关（Partial Autocorrelation Coefficient, PACF）函数图，对于平稳时间序列而言，其自相关或偏自相关系数一般会在某一阶后变为迅速降低为0左右，而非平稳的时间序列的自相关系数一般则是缓慢下降
————————————————
原文链接：https://blog.csdn.net/elite0/article/details/125503379
"""

#### 引入statsmodels和scipy.stats用于画QQ和PP图
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

"""
ARIMA model:
using three parameters: p, d, and q. 
The p parameter represents the number of autoregressive terms, 
the d parameter represents the degree of differencing needed to make the time series stationary, 
and the q parameter represents the number of moving average terms.
SARIMA model:
The seasonal autoregressive order, P, represents the number of seasonal autoregressive terms to include in the model.
The seasonal differencing, D, represents the number of times that the time series is differenced at the seasonal level to remove the seasonal pattern.
The seasonal moving average order, Q, represents the number of seasonal moving average terms to include in the model.
Together, the seasonal and non-seasonal parameters determine the structure of the SARIMA model. The SARIMA model can be particularly useful for time series data that exhibit both seasonal and non-seasonal patterns. The selection of appropriate values for the seasonal and non-seasonal parameters can be determined using statistical methods such as the Akaike Information Criterion (AIC) or the Bayesian Information Criterion (BIC).
"""
def ARIMA_construction(df,p, d, q,predictedDate):
    # ARIMA model with (p, d, q)
    mod = sm.tsa.statespace.SARIMAX(df, trend='c', order=(p, d, q))
    results = mod.fit(disp=False)
    print(results.summary())
    # graphical statistics of model (correlogram = ACF plot)
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()
    # this code requires the fitted forecasts (for accuracy evaluation) to start 01 Jan 1979.
    pred = results.get_prediction(start=predictedDate, dynamic=False)
    pred_ci = pred.conf_int()
    print(pred_ci)
    # this code requires the whole plot to start in 1956 (start year of data)
    ax = df.plot(label='Original data')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.legend()
    plt.show()
    # MSE evaluation
    y_forecasted = pred.predicted_mean
    # y_truth = df['1965-01-01':]
    y_truth = df
    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('MSE of the forecasts is {}'.format(round(mse, 2)))
    # =============================================
    # get forecast 20 steps ahead in future
    pred_uc = results.get_forecast(steps=20)
    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()

    # plotting forecasts ahead
    ax = df.plot(label='Original data')
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast values', title='Forecast plot with confidence interval')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    plt.legend()
    plt.show()
    # ----------------------------------------------

def SARIMA_construction(df,predictedDate,p,d,q,P,D,Q,s):
    # ARIMA model with (p, d, q)
    mod = sm.tsa.statespace.SARIMAX(df, order=(p,d,q),seasonal_order=(P,D,Q,s))
    results = mod.fit(disp=False)
    print(results.summary())
    # graphical statistics of model (correlogram = ACF plot)
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()
    # this code requires the fitted forecasts (for accuracy evaluation) to start 01 Jan 1979.
    pred = results.get_prediction(start=predictedDate, dynamic=False)
    pred_ci = pred.conf_int()
    print(pred_ci)
    # this code requires the whole plot to start in 1956 (start year of data)
    ax = df.plot(label='Original data')
    pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    plt.legend()
    plt.show()
    # MSE evaluation
    y_forecasted = pred.predicted_mean
    # y_truth = df['1965-01-01':]
    y_truth = df
    # Compute the mean square error
    mse = ((y_forecasted - y_truth) ** 2).mean()
    print('MSE of the forecasts is {}'.format(round(mse, 2)))
    # =============================================
    # get forecast 20 steps ahead in future
    pred_uc = results.get_forecast(steps=20)
    # Get confidence intervals of forecasts
    pred_ci = pred_uc.conf_int()

    # plotting forecasts ahead
    ax = df.plot(label='Original data')
    pred_uc.predicted_mean.plot(ax=ax, label='Forecast values', title='Forecast plot with confidence interval')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    plt.legend()
    plt.show()
    # ----------------------------------------------

"""
here is a summary of some of the common methods that can be used to check stationarity in an ARIMA model:
Visual inspection: Plot the time series data and examine it for trends, seasonality, and other patterns.
Augmented Dickey-Fuller (ADF) test: A statistical test that can be used to test for stationarity. The null hypothesis of the test is that the time series is non-stationary.
KPSS test: Another statistical test that can be used to test for stationarity. The null hypothesis of the test is that the time series is stationary.
Moving average: Calculate the moving average of the time series and examine it for changes in mean and variance over time.
Autocorrelation function (ACF) and partial autocorrelation function (PACF) plots: Used to identify the order of differencing required to make the time series stationary.
Variance ratio test: A statistical test that compares the variance of the time series at different lags and can determine if the variance is constant over time.
"""
def stationary_test(series,lags):
    # plot the time series
    plt.plot(series)
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.show()
    # plot ACF and PACF
    plot_acf(series, lags=lags)
    plt.show()
    plot_pacf(series, lags=lags)
    plt.show()

def ts_plot(data, lags=None, title=''):
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    with plt.style.context('bmh'):
        fig = plt.figure(figsize=(10, 8), dpi=300)
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0))

        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))

        data.plot(ax=ts_ax)
        ts_ax.set_title(title + '时序图')
        smt.graphics.plot_acf(data, lags=lags, ax=acf_ax, alpha=0.5)
        acf_ax.set_title('自相关系数')
        smt.graphics.plot_pacf(data, lags=lags, ax=pacf_ax, alpha=0.5)
        pacf_ax.set_title('偏自相关系数')
        sm.qqplot(data, line='s', ax=qq_ax)
        qq_ax.set_title('QQ 图')
        scs.probplot(data, sparams=(data.mean(), data.std()), plot=pp_ax)
        pp_ax.set_title('PP 图')
        plt.tight_layout()
        plt.show()