# _*_ coding: utf-8 _*_
"""
Time:     2021/12/1 9:48
Author:   ChenXin
Version:  V 0.1
File:     learningLine.py
Describe:  Github link: https://github.com/Chen-X666
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, precision_score, accuracy_score, recall_score
from sklearn.model_selection import learning_curve
import seaborn as sns
import pandas as pd

def plot_learning_curve(algo,X_train,X_test,y_train,y_test,score=mean_squared_error):

    df = pd.DataFrame(columns=['num','score', 'score_type'])

    for i in range(10,len(X_train)+10,10):

        algo.fit(X_train[:i],y_train[:i])
        y_train_predict = algo.predict(X_train[:i])
        y_test_predict = algo.predict(X_test)
        # 新插入的行一定要加 index,不然会报错
        df1 = pd.DataFrame([i,(score(y_train[:i],y_train_predict)).round(3),'train']).T
        # 修改df4的column和df3的一致
        df1.columns = df.columns
        # 新插入的行一定要加 index,不然会报错
        df2 = pd.DataFrame([i,(score(y_test,y_test_predict)).round(3),'test']).T
        # 修改df4的column和df3的一致
        df2.columns = df.columns
        # 把两个dataframe合并，需要设置 ignore_index=True
        df = pd.concat([df, df1,df2], ignore_index=True)
    # df = pd.read_csv('score.csv')
    # print(df)
    # print(df.dtypes)
    # df['num'] = df['num'].astype('int')
    #
    # df['score'] = df['score'].astype('float')
    # df.to_csv('score.csv',index=False)
    sns.set()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    palette = sns.xkcd_palette(["windows blue"])
    sns.lineplot(x="num", y="score", data=df,style='类型')
    plt.axis([10,len(X_train)+10,0,1])

    #plt.plot([i for i in range(10,len(X_train)+10,10)], np.sqrt(train_score),label = 'Train')
    #plt.plot([i for i in range(10,len(X_train)+10,10)], np.sqrt(test_score),label = 'Test')
    #plt.legend()

    #plt.axis([10,len(X_train)+10,0,1])
    plt.xlabel("训练集数据量")
    plt.ylabel("准确度(Accuracy)")
    plt.show()



# __all__ = [
#     'accuracy_score',
#     'adjusted_mutual_info_score',
#     'adjusted_rand_score',
#     'auc',
#     'average_precision_score',
#     'balanced_accuracy_score',
#     'calinski_harabasz_score',
#     'check_scoring',
#     'classification_report',
#     'cluster',
#     'cohen_kappa_score',
#     'completeness_score',
#     'ConfusionMatrixDisplay',
#     'confusion_matrix',
#     'consensus_score',
#     'coverage_error',
#     'dcg_score',
#     'davies_bouldin_score',
#     'DetCurveDisplay',
#     'det_curve',
#     'euclidean_distances',
#     'explained_variance_score',
#     'f1_score',
#     'fbeta_score',
#     'fowlkes_mallows_score',
#     'get_scorer',
#     'hamming_loss',
#     'hinge_loss',
#     'homogeneity_completeness_v_measure',
#     'homogeneity_score',
#     'jaccard_score',
#     'label_ranking_average_precision_score',
#     'label_ranking_loss',
#     'log_loss',
#     'make_scorer',
#     'nan_euclidean_distances',
#     'matthews_corrcoef',
#
#     'multilabel_confusion_matrix',
#     'mutual_info_score',
#     'ndcg_score',
#     'normalized_mutual_info_score',
#     'pair_confusion_matrix',
#     'pairwise_distances',
#     'pairwise_distances_argmin',
#     'pairwise_distances_argmin_min',
#     'pairwise_distances_chunked',
#     'pairwise_kernels',
#     'plot_confusion_matrix',
#     'plot_det_curve',
#     'plot_precision_recall_curve',
#     'plot_roc_curve',
#     'PrecisionRecallDisplay',
#     'precision_recall_curve',
#     'precision_recall_fscore_support',
#     'precision_score',
#     'r2_score',
#     'rand_score',
#     'recall_score',
#     'RocCurveDisplay',
#     'roc_auc_score',
#     'roc_curve',
#     'SCORERS',
#     'silhouette_samples',
#     'silhouette_score',
#     'top_k_accuracy_score',
#     'v_measure_score',
#     'zero_one_loss',
#     'brier_score_loss',
# ]