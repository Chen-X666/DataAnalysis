
#coding:gbk,
#ʹ��Pandas��read_csv��read_fwf��read_table��ȡ����

import pandas as pd  # ����Pandas��

csv_data = pd.read_csv('csv_data.csv', names=['col1', 'col2', 'col3', 'col4', 'col5'])  # ��ȡcsv����
print(csv_data)  # ��ӡ�������
print (type(csv_data))


fwf_data = pd.read_fwf('fwf_data', widths=[5, 3, 6, 6],
                       names=['col1', 'col2', 'col3', 'col4'])  # ��ȡcsv����
print(fwf_data)  # ��ӡ�������

table_data = pd.read_table('table_data.txt', sep=';',
                           names=['col1', 'col2', 'col3', 'col4', 'col5'])  # ��ȡtxt����
print(table_data)  # ��ӡ�������


