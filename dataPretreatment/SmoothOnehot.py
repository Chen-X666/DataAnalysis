# _*_ coding: utf-8 _*_
"""
Time:     2021/11/1 15:13
Author:   ChenXin
Version:  V 0.1
File:     SmoothOnehot.py
Describe:  Github link: https://github.com/Chen-X666
"""
import torch
import pandas as pd
import numpy as np
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method

    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))    # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)    # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(true_labels, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)     # 必须要torch.LongTensor()
    return true_dist

if __name__ == '__main__':
    df = pd.read_csv('标注集.csv',encoding='GBK')
    df = df.to_list()
    smooth_one_hot(df)

