import warnings
import arch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", ValueWarning)
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

def mutil_linear_regression(X,y,start_data):
    """
    Function: Function Perform multiple linear regression using Ordinary Least Squares (OLS) method.
    Args:
        X (pandas.DataFrame): input features as a pandas dataframe.
        y (pandas.Series): target variable as a pandas series.
        start_data (str): start date for the x-axis in the plot.
    Returns:
        model: trained OLS model object.
    """
    # add a constant to the predictors
    X = sm.add_constant(X)

    # fit the model using ordinary least squares (OLS)
    model = sm.OLS(y, X).fit()

    yFit = model.fittedvalues  # predicted values of y using the model
    print(model.summary())  # print the summary of the regression analysis
    print("\nOLS model: Y = b0 + b1*X + ... + bm*Xm")
    print('Parameters: ', model.params)  # print the coefficients of the fitted model

    # Calculate MSE using the fitted model and the original data
    y_pred = model.predict(sm.add_constant(X))
    mse = mean_squared_error(y, y_pred).round(2)
    print(f"MSE is {mse}")

    # plot actual vs predicted values
    fig, ax = plt.subplots()
    ax.scatter(y, model.predict(), edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label="OLS line")
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title("Regression: actual vs. predicted with OLS line")
    plt.legend()
    plt.show()

    # plot original data points, the fitted curve and confidence intervals
    x1 = pd.date_range(start=start_data, periods=len(X), freq='M')
    prstd, ivLow, ivUp = wls_prediction_std(model)  # returns standard deviation and confidence intervals
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x1, y, 'o', label="original")  # experimental data (original data+error)
    ax.plot(x1, yFit, 'r-', label="OLS line(Predictions)")  # fitted data
    ax.plot(x1, ivUp, '--', color='orange', label=f"Confident interval\nMSE={mse}")  # confidence interval upper bound
    ax.plot(x1, ivLow, '--', color='orange')  # confidence interval lower bound
    ax.legend(loc='best', fontsize=18)  # show the legend
    ax.set_title("Regression: Original and predicted point with confident interval")
    plt.xlabel('year')
    plt.ylabel('y_value(GMAK)')
    plt.show()
    return model
def plotData(df):
    """
    Function: Plots a time-series dataset as a line chart.
    Parameters:
        df (pandas DataFrame): A DataFrame with time-series data, where each row represents a timestamp and each column represents a variable of interest.
    Returns:
        None
    Outputs:
        - A line chart of the input DataFrame, with the x-axis representing the timestamps and the y-axis representing the values of the variables.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot_date(df.index, df, '-')
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_xlabel('year')
    ax.set_ylabel('R(i,t)-Indonesia short-term interest rate ')
    plt.tight_layout()
    plt.legend()
    plt.show()


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
    print(ts_test_output)
    # Check if the test rejects the null hypothesis and concludes that the series is stationary
    if result[1] <= cutoff:
        print(u"Rejected the original hypothesis that the data have no unit root and the series is smooth.")
    else:
        print(u"Cannot rejected the original hypothesis that the data has a unit root and the data is non-stationary.")
    print('Meanwhile, If the p-value is less than the significance level (usually 0.05) then we can reject the null hypothesis and conclude that the time series is stationary.')

def AR_model(df,p):
    model = AutoReg(df, lags=p)
    model_fit = model.fit()
    # Print the summary of the model
    print(f"AR({p}) model summary:")
    print(model_fit.summary())
    print("\n")
    # Plot the original time series and the fitted values
    plt.figure()
    plt.plot(df, label="Original Data")
    plt.plot(model_fit.fittedvalues, label=f"Fitted AR({p}) Data")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"AR({p}) Model")
    plt.show()

def ARMA_model(df,p,q):
    model = ARIMA(df, order=(p, 0, q))
    results = model.fit()
    print(f"ARMA({p}, {q}) Model Summary:")
    print(results.summary())

    # Plot the original time series and the fitted values
    plt.figure()
    plt.plot(df, label="Original Data")
    plt.plot(results.fittedvalues, label=f"Fitted ARMA({p}, {q}) Data")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"ARMA({p}, {q}) Model")
    plt.show()

if __name__ == '__main__':
    dataset = pd.read_excel('dataset.xlsx', sheet_name='Sheet1', parse_dates=['time'],
                              squeeze=True, index_col=0)
    dataset = dataset.dropna()
    # dataset = dataset [dataset.index >= '2000-01-01']
    # print(dataset)
    # X = dataset[['USinterest', 'VIX','EPU']]  # predictors
    # # dataset.plt()
    # X = dataset[['USinterest']]  # predictors
    # y = dataset['indonesia']  # response variable
    dataset = dataset['indonesia']
    # model = mutil_linear_regression(X=X, y=y, start_data=dataset.index.min())
    stationary_ADF_test(series=dataset)
    # Compute the first difference
    plotData(df=dataset)
    first_difference = (dataset - dataset.shift(1)).dropna(inplace=False)
    stationary_ADF_test(series=first_difference)
    # Fit AR(p) models with different orders
    p_values = [1, 2, 3]  # Orders for AR models
    q_values = [1, 2, 3]  # Orders for ARMA models

    # Initialize summary table
    summary_table = pd.DataFrame(
        columns=['Model', 'Coefficients', 'Standard Errors', 'AIC', 'BIC', 'Ljung-Box Q-Stat', 'Ljung-Box P-Value'])
    # Fit AR(p) models for p=1,2,3
    for p in range(1, 4):
        model = AutoReg(first_difference, lags=p)
        results = model.fit()
        print(f"AR({p}) Model Summary:")
        print(results.summary())

        # Calculate the Ljung-Box Q test for the first 12 autocorrelations in the residual series
        lb_test = acorr_ljungbox(results.resid, lags=12, return_df=True)

        # Append model information to the summary table
        summary_table = summary_table.append({
            'Model': f'AR({p})',
            'Coefficients': results.params,
            'Standard Errors': results.bse,
            'AIC': results.aic,
            'BIC': results.bic,
            'Ljung-Box Q-Stat': lb_test['lb_stat'].values,
            'Ljung-Box P-Value': lb_test['lb_pvalue'].values
        }, ignore_index=True)

    print("Summary Table:")
    print(summary_table)

    # Ljung_Box_Q_Stat = list(summary_table['Ljung-Box P-Value'].values)
    # Ljung_Box_Q_Stat = np.round(Ljung_Box_Q_Stat, decimals=3)
    # pd.DataFrame(Ljung_Box_Q_Stat).to_csv('Ljung-Box P-Value.csv',encoding='utf-8')
    # # Initialize summary table
    # summary_table = pd.DataFrame(
    #     columns=['Model', 'Coefficients', 'Standard Errors', 'AIC', 'BIC', 'Ljung-Box Q-Stat', 'Ljung-Box P-Value'])
    #
    # # Fit ARMA(p, q) models for p=1,2,3 and q=1,2,3
    # for p in range(1, 4):
    #     for q in range(1, 4):
    #         model = ARIMA(first_difference, order=(p, 0, q))
    #         results = model.fit()
    #         print(f"ARMA({p}, {q}) Model Summary:")
    #         print(results.summary())
    #
    #         # Calculate the Ljung-Box Q test for the first 12 autocorrelations in the residual series
    #         lb_test = acorr_ljungbox(results.resid, lags=12, return_df=True)
    #
    #         # Append model information to the summary table
    #         summary_table = summary_table.append({
    #             'Model': f'ARMA({p}, {q})',
    #             'Coefficients': results.params,
    #             'Standard Errors': results.bse,
    #             'AIC': results.aic,
    #             'BIC': results.bic,
    #             'Ljung-Box Q-Stat': lb_test['lb_stat'].values,
    #             'Ljung-Box P-Value': lb_test['lb_pvalue'].values
    #         }, ignore_index=True)
    #
    # print("Summary Table:")
    # print(summary_table)
    Ljung_Box_Q_Stat = list(summary_table['Ljung-Box Q-Stat'].values)
    Ljung_Box_P_Value = list(summary_table['Ljung-Box P-Value'].values)
    Ljung_Box_Q_Stat = np.round(Ljung_Box_Q_Stat, decimals=3)
    Ljung_Box_P_Value = np.round(Ljung_Box_P_Value, decimals=3)
    Ljung_Box = []
    for i in range(0,9,1):
        Ljung_Box.append(Ljung_Box_Q_Stat[i])
        Ljung_Box.append(Ljung_Box_P_Value[i])

    pd.DataFrame(Ljung_Box).to_csv('Ljung-Box Q-Stat.csv',encoding='utf-8')
    #
    #
    # model = arch.arch_model(first_difference, vol='Garch', p=1)
    # model_fit = model.fit()
    # print(model_fit.summary())