# _*_ coding: utf-8 _*_
"""
Time:     2022/2/11 12:41
Author:   ChenXin
Version:  V 0.1
File:     ExponentialSmoothing.py
Describe:  Github link: https://github.com/Chen-X666
"""
from dataprep.eda import create_report
if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('data/video_table.csv', encoding='utf-8')

    create_report(df)