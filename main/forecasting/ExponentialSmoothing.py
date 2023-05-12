import datetime
from itertools import product
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
sns.set()

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
    threshold = 0.9

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

def trend_seasonal_decompose(df):
    """
    Function: Decomposes a time-series dataset into its trend, seasonal, and residual components using the additive model.
    Parameters:
        df (pandas DataFrame): A DataFrame with time-series data, where each row represents a timestamp and each column represents a variable of interest.
    Returns:
        statsmodels.tsa.seasonal.SeasonalResult: A seasonal decomposition object containing the extracted trend, seasonal, and residual components of the input time series.
    Outputs:
        - A plot of the decomposed time series, showing the original data, the extracted trend, seasonal, and residual components.
    """
    # decompose time series data to check for trend and seasonality
    result = seasonal_decompose(df, model='additive')
    # plot decomposed time series data
    result.plot()
    plt.show()
    return result

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
    plt.show()

def holt_winter_modelling_in_gridsearch(train_data,test_data,trend=None, seasonal=None, seasonal_periods=None):
    """
    Function: Perform grid search to find the best Holt-Winters model parameters.
    Args:
        train_data (pd.Series): A pandas DataFrame containing the training data.
        test_data (pd.Series): A pandas DataFrame containing the test data.
        trend (str or None): The type of trend component to use. Can be None, 'add', or 'mul'.
        seasonal (str or None): The type of seasonal component to use. Can be None, 'add', or 'mul'.
        seasonal_periods (int or None): The number of periods in each seasonal cycle.
    Returns:
        None
    """
    # set up parameter grid
    param_grid = {'smoothing_level': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1],
                  'smoothing_slope': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1],
                  'smoothing_seasonal': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,0.9,1]}

    # perform grid search
    best_params = None
    best_mse = np.inf
    aics = []

    # evaluate the model for each combination of alpha and beta values
    mse_values = []
    for params in product(*param_grid.values()):
        model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=12)
        fitted_model = model.fit(smoothing_level=params[0], smoothing_trend=params[1],
                                     smoothing_seasonal=params[2])
        predictions = fitted_model.forecast(len(test_data))
        try:
            mse = mean_squared_error(test_data, predictions)
            aic = fitted_model.aic
            aics.append(aic)
        except Exception:
            mse = 1e10
        # record the MSE value for this combination of alpha and beta
        mse_values.append((params[0], params[1], params[2],mse))
        if mse < best_mse:
            best_params = params
            best_mse = mse

    print('Best parameters:', best_params)
    print('Best MSE:', best_mse)
    print(min(aics))

    # initialize and fit the final model with the best parameters
    model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fit = model.fit(smoothing_level=best_params[0], smoothing_trend=best_params[1],
                                 smoothing_seasonal=best_params[2])
    # Make predictions for the test data
    predictions = fit.forecast(len(test_data))

    # Plot the actual and predicted values
    fig = plt.figure(1, figsize=(8, 6))
    plt.title(f'Holt-winter smoothing in predicting on JVZ8 dataset \n MSE={best_mse}')
    plt.plot(train_data.index, train_data.values, label='Training Data')
    plt.plot(test_data.index, test_data.values, label='Test Data')
    plt.plot(predictions.index, predictions.values, label='Predictions')
    plt.xlabel("year")
    plt.ylabel("value")
    plt.legend()
    plt.show()

    # plot a heatmap of the MSE values
    mse_df = pd.DataFrame(mse_values, columns=['alpha', 'beta', 'gamma','mse'])
    # plot the 4D figure
    fig = plt.figure(2, figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    mappable = ax.scatter(mse_df['alpha'].to_list(), mse_df['beta'].to_list(), mse_df['gamma'].to_list(), c=mse_df['mse'].to_list(), cmap='coolwarm')
    clb = plt.colorbar(mappable,ax=ax)
    clb.ax.set_title('MSE \n')
    ax.set_title('MSE of dataset in different parameters')
    ax.set_xlabel('Smoothing Level')
    ax.set_ylabel('Smoothing Slope')
    ax.set_zlabel('Smoothing Seasonal')
    plt.show()

def holt_winter_modelling(train_data, test_data, trend=None, seasonal=None, seasonal_periods=None, smoothing_level=0.5, smoothing_trend=0.5, smoothing_seasonal=0.5):
    """
    Function: Perform a grid search to find the best Holt-Winters parameters for a given dataset.
    Args:
        train_data (pd.Series): Time series data used for training the model.
        test_data (pd.Series): Time series data used for testing the model.
        trend (str or None): Type of trend component. Can be "add" or "mul". Default is None.
        seasonal (str or None): Type of seasonal component. Can be "add" or "mul". Default is None.
        seasonal_periods (int or None): The number of periods in a complete seasonal cycle. Default is None.
    Returns:
        pd.Series: A series of predicted values for the test data.
    """
    # initialize and fit the final model with the best parameters
    model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fit = model.fit(smoothing_level=smoothing_level, smoothing_trend=smoothing_trend,
                    smoothing_seasonal=smoothing_seasonal)
    original_dataset = pd.concat([train_data, test_data])
    # Make predictions for the test data
    predictions = fit.predict(start=original_dataset.index.max()+datetime.timedelta(days=1),end=datetime.datetime(2023, 12,1))
    # Plot the actual and predicted values
    fig = plt.figure(1, figsize=(8, 6))
    plt.plot(original_dataset.index, original_dataset.values, label='Original Data')
    plt.plot(predictions.index, predictions.values, label='Predictions')
    plt.xlabel("year")
    plt.ylabel("value")
    plt.legend()
    plt.show()
    return predictions

def save_predicted_result(file_name,results):
    """
    Function: Save the predicted results to an Excel file.
    Args:
        file_name (str): The name of the Excel file to save the predicted results to.
        results (pandas.DataFrame): The predicted results to save.
    Returns:
        None
    """
    # Create a DataFrame from the predicted results
    df = pd.DataFrame(data={"date": results.index, "value": results.values})
    # create a new sheet in the existing Excel file with the DataFrame
    with pd.ExcelWriter(file_name, engine='openpyxl', mode='a') as writer:
        df.to_excel(writer, sheet_name='Predicted Data',index=False)

if __name__ == '__main__':
    # load dataset
    data_K550  = pd.read_excel('K550data_33662797_32741154_33991383.xlsx',sheet_name='K55O',parse_dates=['CDID'], squeeze=True,index_col=0)
    data_JVZ8 = pd.read_excel('JVZ8data_33662797_32741154_33991383.xlsx', sheet_name='JVZ8', parse_dates=['CDID'], squeeze=True, index_col=0)
    data_GMAA = pd.read_excel('GMAAdata_33662797_32741154_33991383.xlsx', sheet_name='GMAA', parse_dates=['CDID'], squeeze=True, index_col=0)
    data_GMAK = pd.read_excel('GMAKdata_33662797_32741154_33991383.xlsx', sheet_name='GMAK', parse_dates=['CDID'], squeeze=True, index_col=0)

    # # abnormal data detection
    abnormal_data_dectection(data_GMAA)
    abnormal_data_dectection(data_GMAK)

    # delete abnormal data
    data_GMAA = data_GMAA [data_GMAA.index < '2020 MAR']
    data_GMAK = data_GMAK [data_GMAK.index < '2020 MAR']
    #
    # # decompose seasonal and trend
    trend_seasonal_decompose(data_K550)
    trend_seasonal_decompose(data_GMAK)
    trend_seasonal_decompose(data_GMAA)
    trend_seasonal_decompose(data_GMAK)

    # # K550 dataset
    # # split dataset into training data and test data according 8:2
    # split_date = data_K550.index[int(len(data_K550.index) * 0.8)]
    # train_data = data_K550.loc[data_K550.index <= split_date]
    # test_data = data_K550.loc[data_K550.index > split_date]
    # print(f'The split datas are train data <= {split_date} and test data > {split_date}')
    # print( f'The length of data_K550 training data and test data are {len(train_data)} and {len(test_data)}')
    # # use grid-search in modelling to find the optimal parameters
    # holt_winter_modelling_in_gridsearch(train_data=train_data,test_data=test_data,trend='add', seasonal='add', seasonal_periods=12)
    # # forcaste using optimal parameter
    # results = holt_winter_modelling(train_data=train_data, test_data=test_data, trend='add', seasonal='add', seasonal_periods=12,
    #                       smoothing_level=0.5, smoothing_trend=1, smoothing_seasonal=0.5)
    # # save_predicted_result(file_name="K550data.xlsx",results=results)
    #

    # JVZ8 dataset
    # split dataset into training data and test data according 8:2
    split_date = data_JVZ8.index[int(len(data_JVZ8.index) * 0.8)]
    train_data = data_JVZ8.loc[data_JVZ8.index <= split_date]
    test_data = data_JVZ8.loc[data_JVZ8.index > split_date]
    print(f'The split datas are train data <= {split_date} and test data > {split_date}')
    print( f'The length of data_K550 training data and test data are {len(train_data)} and {len(test_data)}')
    # use grid-search in modelling to find the optimal parameters
    holt_winter_modelling_in_gridsearch(train_data=train_data,test_data=test_data,trend='mul', seasonal='add', seasonal_periods=12)
    results = holt_winter_modelling(train_data=train_data, test_data=test_data, trend='mul', seasonal='add', seasonal_periods=12,
                           smoothing_level=0.8, smoothing_trend=0.7, smoothing_seasonal=0.8)
    #save_predicted_result(file_name="JVZ8data.xlsx", results=results)

    # GMAA dataset
    # split dataset into training data and test data according 8:2
    split_date = data_GMAA.index[int(len(data_GMAA.index) * 0.8)]
    print(split_date)
    train_data = data_GMAA.loc[data_GMAA.index <= split_date]
    test_data = data_GMAA.loc[data_GMAA.index > split_date]
    print(f'The split datas are train data <= {split_date} and test data > {split_date}')
    print( f'The length of data_K550 training data and test data are {len(train_data)} and {len(test_data)}')
    # use grid-search in modelling to find the optimal parameters
    holt_winter_modelling_in_gridsearch(train_data=train_data, test_data=test_data, trend='mul', seasonal='add', seasonal_periods=12)
    # forcaste using optimal parameter
    results = holt_winter_modelling(train_data=train_data, test_data=test_data, trend='mul', seasonal='add', seasonal_periods=12,
                          smoothing_level=0.6, smoothing_trend=0.1, smoothing_seasonal=0.7)
    #save_predicted_result(file_name="GMAAdata.xlsx", results=results)

    # GMAK dataset
    split_date = data_GMAK.index[int(len(data_GMAK.index) * 0.8)]
    print(split_date)
    train_data = data_GMAK.loc[data_GMAK.index <= split_date]
    test_data = data_GMAK.loc[data_GMAK.index > split_date]
    print(f'The split datas are train data <= {split_date} and test data > {split_date}')
    print( f'The length of data_K550 training data and test data are {len(train_data)} and {len(test_data)}')
    # use grid-search in modelling to find the optimal parameters
    holt_winter_modelling_in_gridsearch(train_data=train_data, test_data=test_data, trend='mul', seasonal='add', seasonal_periods=12)
    # forcaste using optimal parameter
    results = holt_winter_modelling(train_data=train_data, test_data=test_data, trend='mul', seasonal='add', seasonal_periods=12,
                          smoothing_level=0.3, smoothing_trend=0.4, smoothing_seasonal=0.6)
    #save_predicted_result(file_name="GMAKdata.xlsx", results=results)
