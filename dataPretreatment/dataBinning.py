# _*_ coding: utf-8 _*_
"""
Time:     2021/6/20 14:32
Author:   ChenXin
Version:  V 0.1
File:     dataBinning.py
Describe: Write during the internship at Hikvison, Github link: https://github.com/Chen-X666
"""
import numpy as np
import pandas as pd

def dataBinning(data, bins, column, right=True, labels=None):
    data[column] = pd.cut(data[column], bins, right=right, labels=labels)
    return data

# 时间属性拓展
def datetime2int(data):
    '''
    将日期和时间数据拓展出其他属性，例如星期几、周几、小时、分钟等。
    :param data: 数据集
    :return: 拓展后的属性矩阵
    '''
    date_set = [pd.datetime.strptime(dates, '%Y-%m-%d') for dates in
                data['order_date']]  # 将data中的order_date列转换为特定日期格式
    '''date_set = [pd.to_datetime(dates, format='%Y-%m-%d') for dates in
                data['order_date']] '''

    weekday_data = [data.weekday() for data in date_set]  # 周几
    daysinmonth_data = [data.day for data in date_set]  # 当月几号
    month_data = [data.month for data in date_set]  # 月份

    time_set = [pd.datetime.strptime(times, '%H:%M:%S') for times in
                data['order_time']]  # 将data中的order_time列转换为特定时间格式
    second_data = [data.second for data in time_set]  # 秒
    minute_data = [data.minute for data in time_set]  # 分钟
    hour_data = [data.hour for data in time_set]  # 小时

    final_set = [weekday_data, daysinmonth_data, month_data, second_data, minute_data,
                 hour_data]  # 将属性列表批量组合
    final_matrix = np.array(final_set).T  # 转换为数组并转置
    return final_matrix

'''
labels：给分割后的bins打标签，比如把年龄x分割成年龄段bins后，可以给年龄段打上诸如青年、中年的标签。
labels的长度必须和划分后的区间长度相等，比如bins=[1,2,3]，划分后有2个区间(1,2]，(2,3]，
则labels的长度必须为2。如果指定labels=False，则返回x中的数据在第几个bin中（从0开始）
'''
if __name__ == '__main__':
    df = pd.read_excel('typedata.xls')

    print(df)
    # 分箱组数 10到15一组15到20一组其余类推
    bins = [10, 15, 20, 25, 30, 35, 40]
    # 分箱列名
    column = 'AGE'
    a = dataBinning(df, bins, column)
    print(a)
    #datetime2int(df)