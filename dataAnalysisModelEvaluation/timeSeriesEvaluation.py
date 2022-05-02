# _*_ coding: utf-8 _*_
"""
Time:     2022/2/18 10:54
Author:   ChenXin
Version:  V 0.1
File:     timeSeriesEvaluation.py
Describe:  Github link: https://github.com/Chen-X666
"""
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
