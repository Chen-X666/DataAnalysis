# _*_ coding: utf-8 _*_
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn import neighbors
from sklearn.preprocessing import minmax_scale
import seaborn as sns

def MeanValue(data,column):
    missingData = data.loc[:, column].values.reshape(-1, 1)
    imp_mean = SimpleImputer(strategy='mean')
    imp_mean = imp_mean.fit_transform(missingData)
    data.loc[:, column] = imp_mean
    data.info()
    return data

def MedianValue(data,column):
    missingData = data.loc[:, column].values.reshape(-1, 1)
    imp_median = SimpleImputer(strategy='median')
    imp_median = imp_median.fit_transform(missingData)
    data.loc[:, column] = imp_median
    data.info()
    return data

def MostFrequentValue(data,column):
    missingData = data.loc[:, column].values.reshape(-1, 1)
    imp_mostFrequent = SimpleImputer(strategy='most_frequent')
    imp_mostFrequent = imp_mostFrequent.fit_transform(missingData)
    data.loc[:, column] = imp_mostFrequent
    data.info()
    return data

def KNNValue(data):
    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    imputed = imputer.fit_transform(data)
    df_imputed = pd.DataFrame(imputed, columns=data.columns)
    return df_imputed


def plot_KNN(data,null_value_index):
    data = pd.DataFrame(data=data,columns=['x','y'])
    data['default'] = 1
    data.at[null_value_index,'default'] = 0
    print(data)
    sns.set()
    sns.scatterplot(x="x", y="y", data=data,hue ='default')
    plt.show()
