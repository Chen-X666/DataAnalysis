# _*_ coding: utf-8 _*_
"""
Time:     2021/6/23 13:55
Author:   ChenXin
Version:  V 0.1
File:     SMOTE.py
Describe: Github link: https://github.com/Chen-X666
"""
# 样本均衡

from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOTE

def sample_balance(X, y):
    '''
    使用SMOTE方法对不均衡样本做过抽样处理
    :param X: 输入特征变量X
    :param y: 目标变量y
    :return: 均衡后的X和y
    '''
    model_smote = SMOTE()  # 建立SMOTE模型对象
    x_smote_resampled, y_smote_resampled = model_smote.fit_resample(X, y)  # 输入数据并作过抽样处理
    return x_smote_resampled, y_smote_resampled