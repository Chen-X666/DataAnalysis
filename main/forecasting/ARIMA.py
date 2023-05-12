import itertools
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
sns.set()

def stationary_ADF_test(series,cutoff=0.01):
    """
    Performs the Augmented Dickey-Fuller test on a time series to check for stationarity.
    Parameters:
        series (pandas Series): The time series data to be tested for stationarity.
        cutoff (float): The significance level for the test. Default is 0.01.
    Returns:
        None. The function prints the results of the test.
    """
    # plot ACF and PACF
    plot_acf(series, lags=50)
    plt.show()
    plot_pacf(series, lags=50)
    plt.show()
    # run the Augmented Dickey-Fuller test
    result = sm.tsa.stattools.adfuller(series,autolag = 'AIC')
    print(' If the ADF Statistic is less than the critical values, we can reject the null hypothesis and conclude that the time series is stationary.')
    print('ADF Statistic: %f' % result[0])
    # Print the critical values for different significance levels
    ts_test_output = pd.Series(result[0:4],
                               index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in result[4].items():
        ts_test_output['Critical Value (%s)' % key] = value
    # Check if the test rejects the null hypothesis and concludes that the series is stationary
    if result[1] <= cutoff:
        print(u"Rejected the original hypothesis that the data have no unit root and the series is smooth.")
    else:
        print(u"Cannot rejected the original hypothesis that the data has a unit root and the data is non-stationary.")
    print('Meanwhile, If the p-value is less than the significance level (usually 0.05) then we can reject the null hypothesis and conclude that the time series is stationary.')

def white_noise_test(ts_diff):
    """
       Function: Test for white noise using the Ljung-Box test on the first-order differenced time series.
       Parameters:
           ts_diff (pandas.Series): A time series after differencing.
       Returns:
           None: Prints the results of the white noise test.
       """
    # Perform white noise test on the first-order differenced time series
    noiseRes = acorr_ljungbox(ts_diff, lags=22)
    # Print the results of the white noise test
    print('One-order difference time series white noise test results:')
    print('Lag | Q-statistic | p-value')
    print(noiseRes)

def stationary_MS_test(ts):
    """
    Function: Performs a stationary test using the mean and standard deviation of a time series.
    Parameters:
        ts (pandas.Series): Time series data to be tested.
    Returns:
        None: The function only plots the original time series, the rolling mean and the rolling standard deviation.
    """
    # Calculate rolling mean and standard deviation
    rol_mean = ts.rolling(window=12, center=False).mean()
    rol_std = ts.rolling(window=12, center=False).std()
    # Plot original time series, rolling mean and rolling standard deviation
    plt.plot(ts, color='blue', label='Original')
    plt.plot(rol_mean, color='red', linestyle='-.', label='Mean')
    plt.plot(rol_std, color='black', linestyle='--', label='Standard Deviation')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.legend(loc='best')
    plt.title('Mean and Standard Deviation')
    plt.show(block=True)

def seasonal_first_diff(series):
    """
    Function: Compute the seasonal first difference of a time series.
    Parameters:
        series (pandas.Series): A time series to compute the seasonal first difference.
    Returns:
        seasonal_first_difference (pandas.Series): The seasonal first difference of the input time series.
    """
    # Compute the first difference
    first_difference = series - series.shift(1)

    # Compute the seasonal first difference
    seasonal_first_difference = first_difference - first_difference.shift(12)

    # Drop the first row containing NaN values due to shifting
    seasonal_first_difference = seasonal_first_difference.dropna(inplace=False)

    return seasonal_first_difference

def SARIMAX_Model(train_data,test_data,order,seasonal_order):
    """
        Fits a seasonal ARIMA model to the training data and generates predictions for the test data.

        Args:
            train_data (pandas.Series): Time series data for training the model.
            test_data (pandas.Series): Time series data for testing the model.
            order (tuple): The (p, d, q) order of the non-seasonal components of the model.
            seasonal_order (tuple): The (P, D, Q, S) order of the seasonal components of the model.

        Returns:
            results (statsmodels.tsa.statespace.sarimax.SARIMAXResultsWrapper): The fitted SARIMAX model results.
        """
    # Create SARIMAX model and fit to training data
    model = sm.tsa.statespace.SARIMAX(train_data, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    # Print summary of model results
    print(results.summary())

    # Generate predictions for test data
    predictions = results.predict(start=test_data.index.min(), end=test_data.index.max())
    print(predictions)

    # Calculate MSE and AIC
    mse = ((predictions - test_data.values.squeeze()) ** 2).mean()
    aic = results.aic

    print(f"Mean squared error (MSE): {mse:.4f}")
    print(f"Akaike Information Criterion (AIC): {aic:.4f}")

    # Generate forecast and confidence interval
    forecast = results.get_prediction(start=test_data.index.min(), end=test_data.index.max(), dynamic=True)
    forecast_ci = forecast.conf_int()
    mte_forecast = forecast.predicted_mean
    mte_pred_concat = pd.concat([mte_forecast, forecast_ci], axis=1)
    mte_pred_concat.columns = [u'Forecasted value', u'Upper bound', u'lower bound']
    mte_pred_concat.head()

    # Plot actual, predicted, and confidence interval values
    fig, ax = plt.subplots(1, figsize=(8, 6))
    plt.plot(train_data.index, train_data.values, label='Training Data')
    plt.plot(test_data.index, test_data.values, label='Test Data')
    plt.plot(predictions.index, predictions.values, label='Predictions')
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='b', alpha=.5)
    plt.title("Predicting in test dataset of GMAK using ARIMA with confidence interval")
    plt.xlabel("year")
    plt.ylabel("value")
    plt.legend()
    plt.show()

    return results

def graph_method_in_modelling(df):
    """
    Function: Plots the autocorrelation and partial autocorrelation functions for a given time series DataFrame.
    Parameters:
        df (pandas.DataFrame): The time series data to be plotted.
    Returns:
        None: Displays the autocorrelation and partial autocorrelation plots.
    """
    # Plot autocorrelation and partial autocorrelation graphs to determine optimal parameters for SARIMA model
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(df.iloc[13:], lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(df.iloc[13:], lags=40, ax=ax2)
    plt.show()

def grid_search_in_modelling(df,order,D):
    """
    Function: Perform a grid search to find the best seasonal ARIMA model based on the given parameters.
    Args:
         df: pandas DataFrame, the time series data to be modeled
         order: tuple, the (p,d,q) order of the ARIMA model
         D: int, the degree of differencing
    Returns:
         wf: pandas DataFrame, a table of the AIC values of all tested models and their corresponding parameter values.
    """
    # Initialize variables
    order = order
    P = Q = range(0, 3)
    pdq_x_PDQs = [(x[0], D, x[1], 12) for x in list(itertools.product(P, Q))]
    a = []
    b = []
    c = []
    d = []
    wf = pd.DataFrame()
    # Loop over all possible seasonal parameters and fit corresponding ARIMA model
    for seasonal_param in pdq_x_PDQs:
        try:
            mod = sm.tsa.statespace.SARIMAX(df, order=order, seasonal_order=seasonal_param,
                                            enforce_stationarity=False, enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{} - AIC:{}'.format(order, seasonal_param, results.aic))
            a.append(order)
            b.append(seasonal_param)
            c.append(results.aic)
        except:
            continue
    # Compile results into a pandas DataFrame and return it
    wf['pdq'] = a
    wf['pdq_x_PDQs'] = b
    wf['aic'] = c
    wf['mse'] = d
    print(wf[wf['aic'] == wf['aic'].min()])
    print(wf[wf['mse'] == wf['mse'].min()])
    return wf

def model_diagnostic(results):
    """
    Function: Perform diagnostic tests on the results of a fitted SARIMA model.
    Parameters:
        results : result of SARIMAX.fit(). The result of the fitted SARIMA model.
    Returns:
        None, The function produces diagnostic plots and a table of the Ljung-Box test results.
    """
    # plot diagnostic plots
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()
    # perform Ljung-Box test
    r, q, p = sm.tsa.acf(results.resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, 26), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    print(table.set_index('lag'))


def model_predict(results, train_data, test_data):
    """
    Function: Use the fitted ARIMA model to make predictions on future data.
    Args:
        results: A fitted ARIMA model obtained from calling the `fit()` method.
        train_data: The training data used to fit the model.
        test_data: The test data on which to make predictions.
    Returns:
        None. Displays a plot of the predicted values and confidence intervals.
    """
    # Combine training and test data to forecast future values
    original_dataset = pd.concat([train_data, test_data])
    # Predict future values for the next 10 years
    forecast = results.get_prediction(start=original_dataset.index.max(), end=datetime(2023, 12, 1), dynamic=True)
    # Get confidence interval for the forecast
    forecast_ci = forecast.conf_int()
    # Get predicted values and confidence intervals
    mte_forecast = forecast.predicted_mean
    mte_pred_concat = pd.concat([mte_forecast, forecast_ci], axis=1)
    mte_pred_concat.columns = [u'Forecasted value', u'Upper bound', u'lower bound']
    print(mte_pred_concat['Forecasted value'])
    # Plot the predicted values and confidence intervals
    ax = original_dataset.plot(label='Original data', figsize=(8, 6))
    forecast.predicted_mean.plot(ax=ax, label='Forecast')
    ax.fill_between(forecast_ci.index,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1], color='g', alpha=.4)
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title("Predicting on dataset of GMAK using ARIMA with confidence interval")
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    data_GMAK = pd.read_excel('GMAKdata_33662797_32741154_33991383.xlsx', sheet_name='GMAK', parse_dates=['CDID'], squeeze=True, index_col=0)
    data_GMAK = data_GMAK[data_GMAK.index < '2020 MAR']
    # Stationary Test
    stationary_MS_test(data_GMAK)
    stationary_ADF_test(data_GMAK)
    # Data Stationary
    data_GMAK_diff = seasonal_first_diff(data_GMAK)
    stationary_MS_test(data_GMAK_diff)
    print(data_GMAK_diff)
    stationary_ADF_test(data_GMAK_diff)
    white_noise_test(data_GMAK_diff)
    graph_method_in_modelling(df=data_GMAK)
    # result: d=1 p=2 q=2 order = (p, d, q)
    grid_search_in_modelling(df=data_GMAK,order=(2, 1, 2),D=1)

    # split the dataset into training data and test data
    split_date = data_GMAK.index[int(len(data_GMAK.index) * 0.8)]
    print(split_date)
    train_data = data_GMAK.loc[data_GMAK.index <= split_date]
    test_data = data_GMAK.loc[data_GMAK.index > split_date]
    # The difference yields d, ACF with a p-order truncated tail and PACF with a q-order trailing tail d=1 p=2 q=2
    order = (2,1,2)
    # P: order of seasonal autoregression (usually not greater than 3) D: number of seasonal differentials (usually not greater than 1)
    # Q: order of seasonal moving average (usually not greater than 3) D=1 P=1 Q=2
    seasonal_order=(0,1,2,12)
    results = SARIMAX_Model(train_data=train_data,test_data=test_data,order=order,seasonal_order=seasonal_order)
    # model_diagnostic(results=results)
    model_predict(results=results,train_data=train_data,test_data=test_data)
