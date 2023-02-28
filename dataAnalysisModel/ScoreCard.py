# _*_ coding: utf-8 _*_
"""
Time:     2022/12/16 17:02
Author:   ChenXin
Version:  V 0.1
Describe:  Github link: https://github.com/Chen-X666
"""
import pandas as pd
import scorecardpy as sc
import numpy as np

def ScoreCard(bins_adj,data_logreg,columns):
    model_sc = sc.scorecard(bins_adj, data_logreg,
                 columns, # The column names in the trained LR
                 points0=600, # Base points
                 odds0=0.01, # Base odds
                 pdo=50) # PDO
    print(model_sc)
    return model_sc

#获取数据的得分
def DataScoreCard(train_noWoE,test_noWoE,model_sc):
    # Applying the credit score. Applies over the original data!
    train_score = sc.scorecard_ply(train_noWoE, model_sc,
                                   print_step=0)
    test_score = sc.scorecard_ply(test_noWoE, model_sc,
                                  print_step=0)
    return train_score,test_score