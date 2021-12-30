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


def plot_learning_curve(algo,X_train,X_test,y_train,y_test,score=mean_squared_error):
    train_score = []
    test_score = []

    for i in range(100,len(X_train)+100,100):

        algo.fit(X_train[:i],y_train[:i])
        y_train_predict = algo.predict(X_train[:i])
        train_score.append((score(y_train[:i],y_train_predict)).round(3))

        y_test_predict = algo.predict(X_test)
        test_score.append((score(y_test,y_test_predict)).round(3))
    sns.set()
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.plot([i for i in range(100,len(X_train)+100,100)], np.sqrt(train_score),label = 'Train')
    plt.plot([i for i in range(100,len(X_train)+100,100)], np.sqrt(test_score),label = 'Test')
    plt.legend()
    plt.axis([100,len(X_train)+100,0,1])
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