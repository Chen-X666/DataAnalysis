from itertools import product
from math import sqrt, log

# create an autocorrelation plot
import pandas as pd
import numpy as np
import seaborn as sns
from pandas import read_excel
from matplotlib import pyplot, pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing

def plotMSEs(labels,MSEs):
    # Calculate the mean squared error for each parameter
    # Plot the mean squared error for each parameter
    plt.plot(labels,MSEs)
    plt.xlabel('Smoothing parameter (alpha)')
    plt.ylabel('Mean squared error')
    plt.title('Grid search results')
    plt.show()

# Simple Exponential Smoothing
def SESmooth(train,test):
    param_grid = np.arange(0.1, 2,0.1)
    MSEs = []
    train.plot(legend=True,color='blue',title='Simple Exponential Smoothing',label='Ture')
    test.plot(color='blue')
    color = ['y','k','g','r','c','m','y','k','g','r','c','m','y','k','g','r','c','m','y','k','g','r','c','m','y','k','g','r','c','m']
    color_i = 0
    for alpha in param_grid:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=round(alpha,2))
        mse = mean_squared_error(ses_model.fittedvalues, train)
        MSEs.append(mse)
        y_pred = ses_model.forecast(len(test)).rename()
        print(y_pred)
        y_pred.plot(legend=True,color=color[color_i],label='alpha={alpha} MSE={MSE}'.format(alpha=round(alpha,2),MSE=round(mse, 2)))
        ses_model.fittedvalues.plot(color=color[color_i])
        color_i = color_i + 1
    plt.show()
    plotMSEs(param_grid,MSEs)


#Holt’s linear Exponential Smoothing
"""
exponential: This parameter controls the degree of smoothing applied to the trend component of the model. A higher value of exponential will give more weight to recent observations, making the trend more responsive to short-term changes in the data. A lower value of exponential will give less weight to recent observations, making the trend smoother and less responsive to short-term changes. You can experiment with different values of exponential and evaluate the accuracy of the resulting forecasts to determine the optimal value.
damped: This parameter controls the rate at which the trend component decays over time. A higher value of damped will cause the trend to approach its long-term average more quickly, which can be useful if you expect the trend to level off or reverse direction in the future. A lower value of damped will cause the trend to continue growing or declining indefinitely. You can experiment with different values of damped and evaluate the accuracy of the resulting forecasts to determine the optimal value.
optimized: This parameter controls whether the smoothing parameters of the model should be optimized using maximum likelihood estimation. When optimized=True, the function will estimate the optimal values for the smoothing parameters based on the training data, which can improve the accuracy of the forecasts. However, this can be computationally expensive, especially for large datasets. You can experiment with setting optimized to True or False and evaluate the accuracy of the resulting forecasts to determine whether optimization is necessary.
"""
def HLESmooth(train_data,test_data,exponential=False,damped=False):
    # Define the range of smoothing parameters to search
    alpha_range = np.linspace(0, 1, 11)
    beta_range = np.linspace(0, 1, 11)
    print(beta_range)
    print(alpha_range)
    params = product(alpha_range, beta_range)

    # Initialize the best MSE and best set of parameters
    best_mse = np.inf
    best_params = None

    # evaluate the model for each combination of alpha and beta values
    mse_values = []

    # Loop through all possible sets of parameters and calculate the MSE
    for param in params:
        try:
            model = Holt(train_data, exponential=exponential,damped=damped)
            fit = model.fit(smoothing_level=param[0], smoothing_trend=param[1],optimized=True)
            predictions = fit.forecast(len(test_data))
            mse = np.mean((predictions - test_data.values) ** 2)
            # record the MSE value for this combination of alpha and beta
            mse_values.append((param[0], param[1], mse))
            if mse < best_mse:
                best_mse = mse
                best_params = param
        except:
            continue

    print('The Best MSE:', best_mse)  # 获得交叉检验模型得出的最优得分,默认是R方
    print('Best Parameters:', best_params)  # 获得交叉检验模型得出的最优参数
    # Fit the Holt's linear exponential smoothing model to the training data using the best set of parameters
    model = ExponentialSmoothing(train_data, exponential=exponential,damped=damped)
    fit = model.fit(smoothing_level=best_params[0], smoothing_trend=best_params[1],optimized=True)

    # Make predictions for the test data
    predictions = fit.forecast(len(test_data))

    # Plot the actual and predicted values
    plt.plot(train_data.index, train_data.values, label='Training Data')
    plt.plot(test_data.index, test_data.values, label='Test Data')
    plt.plot(predictions.index, predictions.values, label='Predictions')
    plt.legend()
    plt.show()

    # plot a heatmap of the MSE values
    mse_df = pd.DataFrame(mse_values, columns=['alpha', 'beta', 'mse'])
    print(mse_df)
    print(mse_df)
    mse_pivot = mse_df.pivot('alpha', 'beta', 'mse')
    sns.heatmap(mse_pivot, cmap='coolwarm', annot=True, fmt=".1f")
    plt.title('MSE values for Holt Linear model')
    plt.xlabel('beta')
    plt.ylabel('alpha')
    plt.show()

    # create a 3D plot of the MSE values
    mse_array = np.array(mse_values)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mse_array[:, 0], mse_array[:, 1], mse_array[:, 2], cmap='coolwarm')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    ax.set_zlabel('MSE')
    plt.title('MSE values for Holt Linear model')
    plt.show()


#Holt-Winter
def Holt_Winter(train_data,test_data,trend=None, seasonal=None, seasonal_periods=None,damped=False):
    # set up parameter grid
    param_grid = {'smoothing_level': [0.1, 0.3, 0.5, 0.7, 0.9,1],
                  'smoothing_slope': [0.1, 0.3, 0.5, 0.7, 0.9,1],
                  'smoothing_seasonal': [0.1, 0.3, 0.5, 0.7, 0.9,1]}

    # perform grid search
    best_params = None
    best_mse = np.inf

    # evaluate the model for each combination of alpha and beta values
    mse_values = []
    for params in product(*param_grid.values()):
        model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=12)
        fitted_model = model.fit(smoothing_level=params[0], smoothing_trend=params[1],
                                     smoothing_seasonal=params[2])
        predictions = fitted_model.forecast(len(test_data))
        mse = np.sqrt(mean_squared_error(test_data, predictions))
        # record the MSE value for this combination of alpha and beta
        mse_values.append((params[0], params[1], params[2],mse))
        if mse < best_mse:
            best_params = params
            best_mse = mse

    print('Best parameters:', best_params)
    print('Best RMSE:', best_mse)

    # initialize and fit the final model with the best parameters
    model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods)
    fit = model.fit(smoothing_level=best_params[0], smoothing_trend=best_params[1],
                                 smoothing_seasonal=best_params[2])
    # Make predictions for the test data
    predictions = fit.forecast(len(test_data))


    # Plot the actual and predicted values
    plt.plot(train_data.index, train_data.values, label='Training Data')
    plt.plot(test_data.index, test_data.values, label='Test Data')
    plt.plot(predictions.index, predictions.values, label='Predictions')
    plt.legend()
    plt.show()

    # plot a heatmap of the MSE values
    mse_df = pd.DataFrame(mse_values, columns=['alpha', 'beta', 'gamma','mse'])
    print(mse_df)
    print(mse_df['alpha'].to_list())
    # plot the 4D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mappable = ax.scatter(mse_df['alpha'].to_list(), mse_df['beta'].to_list(), mse_df['gamma'].to_list(), c=mse_df['mse'].to_list(), cmap='coolwarm')
    clb = plt.colorbar(mappable,ax=ax)
    clb.ax.set_title('MSE \n')
    ax.set_xlabel('Smoothing Level')
    ax.set_ylabel('Smoothing Slope')
    ax.set_zlabel('Smoothing Seasonal')
    plt.show()