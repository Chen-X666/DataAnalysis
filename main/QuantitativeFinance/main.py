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

def abnormal_data_dectection(df):
    """
    Function: Identifies outliers in a time-series dataset using a moving average approach.
    Parameters:
        df (pandas DataFrame): A DataFrame with time-series data, where each row represents a timestamp and each column represents a variable of interest.
    Returns:
        pandas DataFrame: A new DataFrame with the outliers removed, obtained by dropping the rows in the input DataFrame that contain the identified outliers.
    Outputs:
        - A plot of the input DataFrame with outliers marked, showing the original data, the moving average, and the identified outliers.
        - A printout of the new DataFrame without outliers.
    """
    # Define the window size for the moving average
    window_size = 12

    # Calculate the moving average using a rolling window
    moving_avg = df.rolling(window_size).mean()
    # Calculate the moving average and standard deviation

    # Define the threshold for outlier detection
    threshold = 0.8

    # Identify the outliers
    outliers = []
    for i in range(window_size - 1, len(df)):
        diff = abs(df.values[i] - moving_avg.values[i])
        if diff > threshold * moving_avg.values[i]:
            outliers.append(i)
            print(f"Outlier found: Date: {df.index[i]}, Value: {df.values[i]}")

    # Create a new dataframe with the outliers removed
    new_df = df.drop(df.index[outliers])
    # Plot the time-series dataset with outliers marked
    plt.plot(df.index, df.values,label='original data')
    plt.plot(moving_avg.index, moving_avg.values, color='red',label='moving avg')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Time-Series Dataset with Outliers Marked')
    for i in outliers:
        plt.plot(df.index[i], df.values[i], 'ro')
    plt.show()
    plt.legend(loc='upper left')
    # Print the new dataframe without outliers
    print("\nNew Dataset without outliers:\n", new_df)
    return new_df

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
    ax.set_ylabel('value')
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
    # Check if the test rejects the null hypothesis and concludes that the series is stationary
    if result[1] <= cutoff:
        print(u"Rejected the original hypothesis that the data have no unit root and the series is smooth.")
    else:
        print(u"Cannot rejected the original hypothesis that the data has a unit root and the data is non-stationary.")
    print('Meanwhile, If the p-value is less than the significance level (usually 0.05) then we can reject the null hypothesis and conclude that the time series is stationary.')

if __name__ == '__main__':
    dataset = pd.read_excel('dataset.xlsx', sheet_name='Sheet1', parse_dates=['time'],
                              squeeze=True, index_col=0)
    dataset = dataset.dropna()
    dataset = dataset [dataset.index >= '2000-01-01']
    # print(dataset)
    # X = dataset[['USinterest', 'VIX','EPU']]  # predictors
    # # dataset.plt()
    # X = dataset[['USinterest']]  # predictors
    # y = dataset['indonesia']  # response variable
    dataset = dataset['indonesia']
    # model = mutil_linear_regression(X=X, y=y, start_data=dataset.index.min())
    stationary_ADF_test(series=dataset)