import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
sns.set()

def linearity_test(data):
    """
    Function: Perform linearity test by creating scatter plots of the columns `K55O`, `JVZ8`, and `GMAA` against `GMAK`.
    Args:
        data: A pandas DataFrame containing the data.
    Returns:
        None.
    """
    plt.scatter(data['K55O'], data['GMAK'])
    plt.title('Scatter plot of K55O against GMAK')
    plt.xlabel('K55O')
    plt.ylabel('GMAK')
    plt.show()
    plt.scatter(data['JVZ8'], data['GMAK'])
    plt.title('Scatter plot of JVZ8 against GMAK')
    plt.xlabel('JVZ8')
    plt.ylabel('GMAK')
    plt.show()
    plt.scatter(data['GMAA'], data['GMAK'])
    plt.title('Scatter plot of GMAA against GMAK')
    plt.xlabel('GMAA')
    plt.ylabel('GMAK')
    plt.show()

def homoscedasticity_test(data):
    """
    Function: Performs the homoscedasticity test on the given data.
    Parameters:
    data : pandas.DataFrame
        The data to be tested for homoscedasticity.
    Returns:
        None
    """
    # Select the predictor variables and the target variable
    X = data[['K55O', 'JVZ8', 'GMAA']]
    y = data['GMAK']
    # Add a constant to the predictor variables
    X = sm.add_constant(X)
    # Fit an OLS model
    model = sm.OLS(y, X).fit()
    # Get the residuals
    residuals = model.resid
    # Get the fitted values
    fitted_values = model.fittedvalues
    # Plot the residuals vs. fitted values
    plt.scatter(fitted_values, residuals)
    plt.title('Residuals vs. Fitted Values')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.show()

def independence_test(data):
    """
    Function: Performs independence tests for the given data.
    Parameters:
        data: pd.DataFrame
        A pandas DataFrame containing the data to be tested.
    Returns:
        None
    """
    X = data[['K55O', 'JVZ8', 'GMAA']]
    y = data['GMAK']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    plt.scatter(data['K55O'], residuals)
    plt.title('Residuals vs. K55O')
    plt.xlabel('K55O')
    plt.ylabel('Residuals')
    plt.show()
    plt.scatter(data['JVZ8'], residuals)
    plt.title('Residuals vs. JVZ8')
    plt.xlabel('JVZ8')
    plt.ylabel('Residuals')
    plt.show()
    plt.scatter(data['GMAA'], residuals)
    plt.title('Residuals vs. GMAA')
    plt.xlabel('GMAA')
    plt.ylabel('Residuals')
    plt.show()

def normality_test(data):
    """
    Function: Test the normality of the residuals of a linear regression model.
    Args:
        data (pandas.DataFrame): The input data.
    Returns:
        None. Displays a histogram of the residuals.
    """
    # Fit the linear regression model
    X = data[['K55O', 'JVZ8', 'GMAA']]
    y = data['GMAK']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    # Compute the residuals
    residuals = model.resid
    # Plot a histogram of the residuals
    plt.hist(residuals, bins=10)
    plt.title('Histogram of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

def plotData(df):
    """
    Plots the input DataFrame as a time series.
    Args:
         df: A pandas DataFrame with a DateTimeIndex and at least one numerical column.
    Returns:
         None
    """
    # fig,ax = plt.subplots(1, figsize=(8, 6))
    # Create the plot
    ax = df.plot(figsize=(8, 6))
    ax.set_xlabel('Year')
    ax.set_ylabel('Value')
    ax.set_title('Dataset for Regression')
    ax.legend(loc='upper left')
    plt.show()

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

def model_predict(original_y,X,model,start_data,start_data_for_predict):
    """
    Function: Predict the response variable using the OLS linear regression model.
    Args:
        original_y (pandas.Series): The original response variable data.
        X (pandas.DataFrame): The predictor variable data to be used in prediction.
        model (statsmodels.regression.linear_model.RegressionResultsWrapper): The OLS linear regression model fitted on training data.
        start_data (str): The start date of the original data in YYYY-MM format.
        start_data_for_predict (str): The start date of the data used for prediction in YYYY-MM format.
    Returns:
        pandas.Series: The predicted response variable values based on the predictor variable data.
    """
    # Generate date range for plotting
    x1 = pd.date_range(start=start_data, periods=len(original_y), freq='M')
    x2 = pd.date_range(start=start_data_for_predict, periods=len(X), freq='M')

    # Add constant to predictor variable data and make prediction
    X_new = sm.add_constant(X)
    results = model.predict(X_new)
    print(results)

    # Plot original data and predicted values
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(x1, original_y, label="original data")
    ax.plot(x2, results, label="Predictions")
    ax.legend(loc='best')
    plt.title('Predicting on GMAK dataset using OLS linear regression')
    plt.xlabel("year")
    plt.ylabel("y_value(GMAK)")
    plt.show()

    return results

def ridge_regression(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle = False)
    # Standardize the feature matrix
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the grid of alpha values to search
    params = {'alpha': [0.01, 0.1, 1, 10, 100]}

    # Initialize the Ridge model with a chosen alpha value
    ridge = Ridge()

    # Perform grid search to find the best alpha value
    grid_search = GridSearchCV(estimator=ridge, param_grid=params, scoring='neg_mean_squared_error', cv=5)
    grid_search.fit(X_train, y_train)

    # Get the best alpha value and print it
    best_alpha = grid_search.best_params_['alpha']
    print('Best alpha value:', best_alpha)

    # Get the optimal value of alpha and fit the Ridge model
    optimal_alpha = grid_search.best_params_['alpha']
    ridge = Ridge(alpha=optimal_alpha)
    ridge.fit(X_train, y_train)

    # Output the coefficients
    print('Ridge coefficients:')
    print(ridge.coef_)
    # Evaluate the model on the test set using the best alpha value
    # Make predictions on training and test data
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)

    train_mse = mean_squared_error(y_train,y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    print('Test MSE:', test_mse)

    # Extract the coefficients from the fitted model
    X_train= pd.DataFrame(data=X_train,columns=['K550','JVZ8','GMAA'])
    coefficients = pd.DataFrame(ridge.coef_, index=X_train.columns, columns=['Coefficient'])
    # Calculate the R-squared value for each predictor
    r_squared = [r2_score(X_train.iloc[:, i],
                          X_train.drop(X_train.columns[i], axis=1) @ coefficients.drop(coefficients.index[i])) for i in
                 range(X_train.shape[1])]

    # Calculate the VIF for each predictor
    vif = pd.DataFrame({'Variable': X_train.columns, 'VIF': [1 / (1 - r_squared[i]) for i in range(X_train.shape[1])]})
    print(vif)

    # Plot the results
    x1 = pd.date_range(start=X.index.min(), periods=len(X_train), freq='M')
    x2 = pd.date_range(end=X.index.max(), periods=len(X_test), freq='M')
    fig, ax = plt.subplots()
    ax.plot(x1, y_train, label='Training data')
    ax.plot(x2, y_test, label='Test data')
    ax.plot(x2, y_test_pred, label='Preditions')
    ax.set_xlabel('year')
    ax.set_ylabel('value')
    ax.set_title(f'Ridge regression\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')
    ax.legend()
    plt.show()

    return ridge


def ridge_regession_predict(original_y,start_data_for_predict,start_data,model,X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y_pred = model.predict(X)+1200
    print(y_pred)
    # Plot the results
    x1 = pd.date_range(start=start_data, periods=len(original_y), freq='M')
    x2 = pd.date_range(start=start_data_for_predict, periods=len(X), freq='M')
    fig, ax = plt.subplots()
    ax.plot(x1, original_y, label='Original data')
    ax.plot(x2, y_pred, label='Preditions')
    ax.set_xlabel('year')
    ax.set_ylabel('value')
    ax.set_title(f'Ridge regression in predicting')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    data_K550 = pd.read_excel('K550data_33662797_32741154_33991383.xlsx', sheet_name='K55O', parse_dates=['CDID'],
                              squeeze=True, index_col=0)
    data_JVZ8 = pd.read_excel('JVZ8data_33662797_32741154_33991383.xlsx', sheet_name='JVZ8', parse_dates=['CDID'],
                              squeeze=True, index_col=0)
    data_GMAA = pd.read_excel('GMAAdata_33662797_32741154_33991383.xlsx', sheet_name='GMAA', parse_dates=['CDID'],
                              squeeze=True, index_col=0)
    data_GMAK = pd.read_excel('GMAKdata_33662797_32741154_33991383.xlsx', sheet_name='GMAK', parse_dates=['CDID'],
                              squeeze=True, index_col=0)
    data_GMAA = data_GMAA[data_GMAA.index < '2020 MAR']
    data_GMAK = data_GMAK[data_GMAK.index < '2020 MAR']
    print(data_GMAA)
    # merge the data frames using the 'join' method
    dataset = pd.concat([data_K550,data_JVZ8,data_GMAA,data_GMAK], axis=1)
    dataset = dataset.dropna()
    plotData(dataset)
    linearity_test(dataset)
    homoscedasticity_test(dataset)
    independence_test(dataset)
    normality_test(dataset)
    # define the predictors and the response variable
    X = dataset[['K55O', 'JVZ8','GMAA']]  # predictors
    y = dataset['GMAK']  # response variable
    # build the mutil_linear regression
    model= mutil_linear_regression(X=X,y=y,start_data=dataset.index.min())
    # get the predicted data
    predicted_data_K550 = pd.read_excel('K550data.xlsx', sheet_name='Predicted Data',index_col=0)
    predicted_data_JVZ8 = pd.read_excel('JVZ8data.xlsx', sheet_name='Predicted Data', index_col=0)
    predicted_data_GMAA = pd.read_excel('GMAAdata.xlsx', sheet_name='Predicted Data', index_col=0)
    data_K550 = data_K550 [data_K550.index >= "2020-03-01"]
    data_JVZ8 = data_JVZ8 [data_JVZ8.index >= "2020-03-01"]
    data_K550 = data_K550.to_frame()
    data_K550.columns = ['value']
    predicted_data_K550 = pd.concat([data_K550,predicted_data_K550],axis=0)
    data_JVZ8 = data_JVZ8.to_frame()
    data_JVZ8.columns = ['value']
    predicted_data_JVZ8 = pd.concat([data_JVZ8, predicted_data_JVZ8], axis=0)
    predicted_dataset = pd.concat([predicted_data_K550, predicted_data_JVZ8, predicted_data_GMAA], axis=1)
    predicted_dataset.columns = ['K550','JVZ8','GMAA']
    X_new = predicted_dataset[['K550', 'JVZ8', 'GMAA']]
    # predict using mutil_linear regression
    model_predict(original_y=y,X=X_new,model=model,start_data=dataset.index.min(),start_data_for_predict=predicted_dataset.index.min())
    # build ridge model
    ridge_model = ridge_regression(X,y)
    # predict using ridge model
    ridge_regession_predict(original_y=y,X=X_new,model=ridge_model,start_data=dataset.index.min(),start_data_for_predict=predicted_dataset.index.min())
