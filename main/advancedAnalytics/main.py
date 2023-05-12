# -*- coding: utf-8 -*-
import pandas as pd
from matplotlib import pyplot as plt

import ExploratoryDataAnalysis.dataReview as dataReading
from dataAnalysisModel.classification import RandomForest, XGBoost, LogisticRegression
from sklearn.model_selection import train_test_split # 数据切割
import ExploratoryDataAnalysis.dataPicReading as dataPicReading
from dataPretreatment import SMOTE
from dataPretreatment.dataEncoder import labelEncoder
import seaborn as sns
sns.set(style='whitegrid')
if __name__ == '__main__':
    # read dataset
    df_039 = pd.read_csv('dataset/dataset_039.csv')
    df_110 = pd.read_csv('dataset/dataset_110.csv')
    df_173 = pd.read_csv('dataset/dataset_173.csv')
    df_196 = pd.read_csv('dataset/dataset_196.csv')
    # classified columns
    numeric_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    categorical_columns = ['job','marital','education','default','housing','loan','contact','month','poutcome','y']
    columns = numeric_columns + categorical_columns
    # all datasets
    dataset = []
    dataset.append(df_039)
    dataset.append(df_110)
    dataset.append(df_173)
    dataset.append(df_196)
    # # dataReview
    dataReading.dataSimpleReview(df_039)
    # print(columns)
    # dataReading.relatedAnalysis(df_039,columns=columns)
    # # dataDistribution
    # dataPicReading.dataHistogramReading(data=df_039,columns=numeric_columns,picWidth=2,picHigh=4)
    # dataPicReading.dataBarReading(data=df_039,columns=numeric_columns,picWidth=2,picHigh=4,y_column_name='y')
    sns.countplot(x='y', data=df_196, palette='hls')
    plt.show()
    # one-hot label coding
    df_039 = labelEncoder.labelEncoder(df_039,labels=categorical_columns)
    df_110 = labelEncoder.labelEncoder(df_110, labels=categorical_columns)
    df_173 = labelEncoder.labelEncoder(df_173, labels=categorical_columns)
    df_196 = labelEncoder.labelEncoder(df_196, labels=categorical_columns)
    # Splitting the data set into a training and test set
    X = df_039.iloc[:, 0:-1]
    y = df_039.iloc[:, -1]
    # imbalanced data processing
    X,y = SMOTE.sample_balance(X=X,y=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # # logistic regression model
    lr_model = LogisticRegression.LogisticRegress(X_train,X_test,y_train,y_test)
    # # # decided tree model
    # # # rf_model = RandomForest.decisionTree(X_train,X_test,y_train,y_test)
    # # # ensemble method——XGBoost
    # xgboost_model = XGBoost.XGboost(X_train,X_test,y_train,y_test)