# -*- coding: utf-8 -*-
import pandas as pd
import ExploratoryDataAnalysis.dataReview as dataReading
from dataAnalysisModel.classification import RandomForest
from sklearn.model_selection import train_test_split # 数据切割

from dataPretreatment.dataEncoder import labelEncoder

if __name__ == '__main__':
    # read dataset
    df_039 = pd.read_csv('dataset/dataset_039.csv')
    df_110 = pd.read_csv('dataset/dataset_110.csv')
    df_173 = pd.read_csv('dataset/dataset_173.csv')
    df_196 = pd.read_csv('dataset/dataset_196.csv')
    numeric_columns = ['age','balance','day','duration','campaign','pdays','previous']
    columns = []
    # dataReview
    dataReading.dataSimpleReview(df_039)
    # dataReading.relatedAnalysis(df,columns=columns)
    # dataDistribution
    # dataPicReading.dataHistogramReading(data=df_039,columns=numeric_columns,picWidth=2,picHigh=4)
    # dataPicReading.dataBarReading(data=df_039,columns=numeric_columns,picWidth=2,picHigh=4,y_column_name='y')
    # label code
    category_job, df_039 = labelEncoder.labelEncoder(df_039, 'job')
    category_marital, df_039 = labelEncoder.labelEncoder(df_039, 'marital')
    category_education, df_039 = labelEncoder.labelEncoder(df_039, 'education')
    category_default, df_039 = labelEncoder.labelEncoder(df_039, 'default')
    category_housing, df_039 = labelEncoder.labelEncoder(df_039, 'housing')
    category_contact, df_039 = labelEncoder.labelEncoder(df_039, 'contact')
    category_loan, df_039 = labelEncoder.labelEncoder(df_039, 'loan')
    category_month, df_039 = labelEncoder.labelEncoder(df_039, 'month')
    category_poutcome, df_039 = labelEncoder.labelEncoder(df_039, 'poutcome')
    category_y, df_039 = labelEncoder.labelEncoder(df_039, 'y')
    dataReading.dataSimpleReview(df_039)
    # Splitting the data set into a training and test set
    X = df_039.iloc[:, 0:-1]
    y = df_039.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # lr_model = LogisticRegression.LogisticRegress(X_train,X_test,y_train,y_test)
    rf_model = RandomForest.decisionTree(X_train,X_test,y_train,y_test)