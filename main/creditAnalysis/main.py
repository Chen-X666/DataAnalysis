# _*_ coding: utf-8 _*_
import pandas as pd
import ExploratoryDataAnalysis.dataReview as dataReading
from dataAnalysisModel.classification import RandomForest
from dataAnalysisModelEvaluation import ModelsComparison
from dataPretreatment import MissingValueHanding
from dataPretreatment.dataEncoder.WoE import WoEBin, IVDestribution, IVFiltering, adjustWoEByManual
from sklearn.model_selection import train_test_split # 数据切割
#matplotlib.use('TkAgg')


if __name__ == '__main__':
    df = pd.read_csv('Credit data.csv',encoding='utf-8')
    # SeriousDlqin2yrs,RevolvingUtilizationOfUnsecuredLines,age,NumberOfTime30-59DaysPastDueNotWorse,DebtRatio,MonthlyIncome,NumberOfOpenCreditLinesAndLoans,NumberOfTimes90DaysLate,NumberRealEstateLoansOrLines,NumberOfTime60-89DaysPastDueNotWorse,NumberOfDependents
    columns = df.columns
    columns = ['RevolvingUtilizationOfUnsecuredLines','age','NumberOfTime30-59DaysPastDueNotWorse','DebtRatio','MonthlyIncome','NumberOfOpenCreditLinesAndLoans','NumberOfTimes90DaysLate','NumberRealEstateLoansOrLines','NumberOfTime60-89DaysPastDueNotWorse','NumberOfDependents']
    # 1.1 Carefully pre-process the dataset by considering the following activities
    # 1.1.1 Exploratory data analysis.
    # dataReview
    dataReading.dataSimpleReview(df)
    #dataReading.relatedAnalysis(df,columns=columns)
    # dataDistribution
    # dataPicReading.dataHistogramReading(data=df,columns=columns,picWidth=2,picHigh=5)
    #dataPicReading.dataBarReading(data=df, columns=columns, picWidth=5, picHigh=2)
    # relation analysis
    # dataReading.relatedAnalysis(df,columns=columns)
    df = df[df['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    df = df[df['NumberOfDependents'] < 10]


    # 1.1.2 Missing value handling
    df = df.dropna(subset=["NumberOfDependents"]).reset_index(drop=True)
    MV_data_DF = MissingValueHanding.KNNValue(df)
    # DD_MV_data_NP = TSNE_PCADimensionalDeduction(weight=MV_data_DF.to_numpy(),delimension=2)
    # null_value_index = df[df.isnull().values==True].index
    # MissingValueHanding.plot_KNN(data=DD_MV_data_NP,null_value_index=null_value_index)
    # 1.1.3 Delete the duplicated value
    MV_data_DF = MV_data_DF.drop_duplicates()

    # dataPicReading.dataBoxReading(data=MV_data_DF, columns=columns, picWidth=5, picHigh=2)
    #1.1.4 Outlier detection and treatment
    #OD_data_DF = outlierDection.ZScore_outlier(df=MV_data_DF,columns=['age','MonthlyIncome'],threshold=1).reset_index(drop=True)
    #print(OD_data_DF)
    # OD_data_DF = outlierDection.SVM_outlier(df=OD_data_DF)
    #dataReading.dataSimpleReview(OD_data_DF)
    #dataPicReading.dataHistogramReading(data=MV_data_DF,columns=columns,picWidth=3,picHigh=3)
    #OD_data_df = outlierDection.isolationForest(MV_data_DF)
    #print(OD_data_df)

    # 1.1.5 Splitting the data set into a training and test set
    # x取除了最后一行，y取最后一行
    X = MV_data_DF.iloc[:, 1:]
    y = MV_data_DF.iloc[:, 0]
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    train = pd.concat([Y_train, X_train], axis=1).reset_index(drop=True)
    test = pd.concat([Y_test, X_test], axis=1).reset_index(drop=True)

    # 1.1.3 Binning the variables (if deemed useful)
    # bins the age and monthlyIncome
    train_woe, test_woe, bins = WoEBin(train=train,test=test, yColumnName ='SeriousDlqin2yrs')
    breaks_adj = {
        # Below are the intervals for different bins
        'DebtRatio': [0.1,0.2,0.3,0.4,0.7,1.3],
        'NumberOfTime30-59DaysPastDueNotWorse': [1.0,2.0,3.0,4.0,5.0,6.0,7.0],
        'RevolvingUtilizationOfUnsecuredLines':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1],
        'age':[28.0,48.0,56.0,66.0,80.0],
        'NumberOfTimes90DaysLate':[1.0,2.0,3.0,4.0,5.0,6.0],
        'NumberOfTime60-89DaysPastDueNotWorse': [1.0,2.0,3.0,4.0,5.0],
        #'MonthlyIncome':[1000,5000,10000,12000,15000,20000,30000]
    }
    bins_adj = adjustWoEByManual(train, breaks_adj, yColumnName ='SeriousDlqin2yrs')
    # WoEDistribution(bins_adj)
    IVDestribution(train_woe, bins_adj, yColumnName ='SeriousDlqin2yrs')
    train_woe,test_woe = IVFiltering(train=train_woe, test=test_woe, bins_adj=bins_adj, dropColumns=['NumberOfDependents_woe','NumberOfOpenCreditLinesAndLoans_woe','DebtRatio_woe'])
    X_train_woe = train_woe.iloc[:, 1:]
    y_train_woe = train_woe.iloc[:, 0]
    X_test_woe = test_woe.iloc[:, 1:]
    y_test_woe = test_woe.iloc[:, 0]
    train_no_woe = train.drop(columns=['NumberOfDependents','NumberOfOpenCreditLinesAndLoans','DebtRatio'])
    test_no_woe = test.drop(columns=['NumberOfDependents', 'NumberOfOpenCreditLinesAndLoans', 'DebtRatio'])
    X_train_no_woe = train_no_woe.iloc[:, 1:]
    y_train_no_woe = train_no_woe.iloc[:, 0]
    X_test_no_woe = test_no_woe.iloc[:, 1:]
    y_test_no_woe = test_no_woe.iloc[:, 0]
    train.to_csv('creditData_train.csv',index=False)
    test.to_csv('creditData_test.csv',index=False)
    #X_train_woe, y_train_woe = SMOTE.sample_balance(X=X_train_woe, y=y_train_woe)
    #X_test_woe, y_test_woe = SMOTE.sample_balance(X=X_test_woe, y=y_test_woe)
    # 1.2 Build an intuitive and predictive scorecard using a logistic regression classifier and report the following
    # 1.2.1 The most important variables
    # 1.2.2 The impact of the variables on the target
    # 1.2.3 The performance of the model. Use various performance metrics and discuss their relationship if any.
    # lr_model = logisticRegression.LogisticRegress(X_train_woe,X_test_woe,y_train_woe,y_test_woe)
    rf_model = RandomForest.decisionTree(X_train_no_woe, X_test_no_woe, y_train_no_woe, y_test_no_woe)
    # model_sc = ScoreCard.ScoreCard(bins_adj=bins_adj,data_logreg=lr_model,columns=train_woe.columns[1:])
    # train_score,test_score = ScoreCard.DataScoreCard(train_noWoE=train_no_woe,test_noWoE=test_no_woe,model_sc=model_sc)
    # print(train_score)
    # print(test_score)
    #
    models = [
        {
            'label': 'Logistic Regression',
            'probs': lr_model.predict_proba(X_test_woe)[:,1]
        },
        {
            'label': 'Random Forest',
            'probs': rf_model.predict_proba(X_test_woe)[:,1]
        }
    ]
    ModelsComparison.ROCComparison(models=models,y_test=y_test_woe)
    # 2.2.4 Compare this scorecard with the result of a Random Forest model run over the data. Discuss your results.
    # Why do banks typically use Logistic Regression as their base classifier? What do banks win and lose by doing this?


