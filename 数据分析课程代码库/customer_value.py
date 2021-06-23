#coding:gbk,

# �����
import time  # ����ʱ���
from datetime import datetime

import numpy as np  # ����numpy��
import pandas as pd  # ����pandas��

# ��ȡ����
dtypes = {'ORDERDATE':object, 'ORDERID': object, 'AMOUNTINFO': np.float32}  # ����ÿ����������,�ֵ�
raw_data = pd.read_csv('sales.csv', dtype=dtypes,index_col='USERID')  # ��ȡ�����ļ�
print(raw_data.dtypes) # �ֶ�����
print('-' * 60)

# ��������У��
# ���ݸ���
print('Data Overview:')
print(raw_data.head(4))  # ��ӡԭʼ����ǰ4��
raw_data.tail()  # # ��ӡԭʼ���ݺ��5��
print('-' * 30)
print('Data DESC:')
print(raw_data.describe())  # ��ӡԭʼ���ݻ�����������Ϣ
print('-' * 60)

# ȱʧֵ���
na_cols = raw_data.isnull().any(axis=0)  # �鿴ÿһ���Ƿ����ȱʧֵ
print('NA Cols:')
print(na_cols)  # �鿴����ȱʧֵ����
print('-' * 30)
na_lines = raw_data.isnull().any(axis=1)  # �鿴ÿһ���Ƿ����ȱʧֵ
print('NA Recors:')
print('Total number of NA lines is: {0}'.format(na_lines.sum()))  # �鿴����ȱʧֵ�����ܼ�¼��
print(raw_data[na_lines])  # ֻ�鿴����ȱʧֵ������Ϣ
print('-' * 60)

# �����쳣����ʽת���ʹ���
# �쳣ֵ����
sales_data = raw_data.dropna()  # ��������ȱʧֵ���м�¼
sales_data = sales_data[sales_data['AMOUNTINFO'] > 1]  # �����������<=1�ļ�¼

# ���ڸ�ʽת��
sales_data['ORDERDATE'] = pd.to_datetime(sales_data['ORDERDATE'], format='%Y-%m-%d')  # ���ַ���ת��Ϊ���ڸ�ʽ
print('Sales_data Dtypes:')
print(sales_data.dtypes)  # ��ӡ������ݿ������е���������
print('-' * 60)

# ����ת��
recency_value = sales_data['ORDERDATE'].groupby(sales_data.index).max()  # ����ԭʼ���һ�ζ���ʱ��
frequency_value = sales_data['ORDERID'].groupby(sales_data.index).count()  # ����ԭʼ����Ƶ��
monetary_value = sales_data['AMOUNTINFO'].groupby(sales_data.index).sum()  # ����ԭʼ�����ܽ��


# ����RFM�÷�
# �ֱ����R��F��M�÷�
deadline_date = datetime(2017, 1, 1)  # ָ��һ��ʱ��ڵ㣬���ڼ�������ʱ�����ʱ��ľ���
r_interval = (deadline_date - recency_value).dt.days  # ����R���
# ����
r_score = pd.cut(r_interval, 5, labels=[5, 4, 3, 2, 1])  # ����R�÷�
f_score = pd.cut(frequency_value, 5, labels=[1, 2, 3, 4, 5])  # ����F�÷�
m_score = pd.cut(monetary_value, 5, labels=[1, 2, 3, 4, 5])  # ����M�÷�
print('-' * 60)

# R��F��M���ݺϲ�
rfm_list = [r_score, f_score, m_score]  # ��r��f��m����ά������б�
rfm_cols = ['r_score', 'f_score', 'm_score']  # ����r��f��m����ά������
rfm_pd = pd.DataFrame(np.array(rfm_list).transpose(), dtype=np.int32, columns=rfm_cols,
                      index=frequency_value.index)  # ����r��f��m���ݿ�
print('RFM Score Overview:')
print(rfm_pd.head(4))
print('-' * 60)

# ����RFM�ܵ÷�
# ����һ����Ȩ�÷�
rfm_pd['rfm_wscore'] = rfm_pd['r_score'] * 0.6 + rfm_pd['f_score'] * 0.3 + rfm_pd['m_score'] * 0.1
# ��������RFM���
rfm_pd_tmp = rfm_pd.copy()
rfm_pd_tmp['r_score'] = rfm_pd_tmp['r_score'].astype(np.str)
rfm_pd_tmp['f_score'] = rfm_pd_tmp['f_score'].astype(np.str)
rfm_pd_tmp['m_score'] = rfm_pd_tmp['m_score'].astype(np.str)
# ƴ��
rfm_pd['rfm_comb'] = rfm_pd_tmp['r_score'].str.cat(rfm_pd_tmp['f_score']).str.cat(
    rfm_pd_tmp['m_score'])

# ��ӡ����ͱ�����
# ��ӡ���
print('Final RFM Scores Overview:')
print(rfm_pd.head(4))  # ��ӡ����ǰ4����
print('-' * 30)
print('Final RFM Scores DESC:')
print(rfm_pd.describe())

# ����RFM�÷ֵ������ļ�
rfm_pd.to_csv('sales_rfm_score.csv')  # ��������Ϊcsv

import matplotlib.pyplot as plt

print(sales_data)
print(sales_data.dtypes)
sales_data['month'] = sales_data['ORDERDATE'].dt.month
sales_month = sales_data['AMOUNTINFO '].groupby(sales_data['month']).sum()

# plt.title("2016", fontsize=20)
# squares=[1, 4, 9, 16, 25,45,15,589,26,36,66,88]
# x=[1, 2, 3, 4, 5,6,7,8,9,10,11,12]
# plt.plot(x, squares)
# plt.show()
