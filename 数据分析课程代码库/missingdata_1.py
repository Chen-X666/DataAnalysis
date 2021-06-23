#coding:gbk,
#  ȱʧֵ����

import pandas as pd  # ����pandas��
import numpy as np  # ����numpy��
from sklearn.impute import SimpleImputer
#Imputer������ȱʧֵ����

print('����ȱʧ����')
df = pd.DataFrame(np.random.randn(6, 4),
                  columns=['col1', 'col2', 'col3', 'col4'])  # ����һ������

df.iloc[1:3, 1] = np.nan  # ����ȱʧֵ
df.iloc[4, 3] = np.nan  # ����ȱʧֵ
print(df)

print('\n����Щֵȱʧ')
nan_all = df.isnull()  # ����������ݿ��е�Nֵ
print(nan_all)  # ��ӡ���

print('\n�鿴��Щ��ȱʧ')
nan_col1 = df.isnull().any()  # ��ú���NA����
nan_col2 = df.isnull().all()  # ���ȫ��ΪNA����
print(nan_col1)  # ��ӡ���
print(nan_col2)  # ��ӡ���

print('\n1����ȱʧֵ')
df2 = df.dropna()  # ֱ�Ӷ�������NA���м�¼
#df2 = df.dropna(axis = 1) #ɾ����
print(df2)  # ��ӡ���

print('\n2ʹ��sklearn��ȱʧֵ�滻Ϊ�ض�ֵ')
imp_mean=SimpleImputer(strategy='constant',fill_value=111)
#mean:ƽ��ֵ��median����λ����most_frequent������
imp_mean.fit(df)
print(imp_mean.transform(df))

print('\n3ʹ��pandas��ȱʧֵ�滻Ϊ�ض�ֵ')
nan_result_pd1 = df.fillna(method='backfill')  # �ú����ֵ�滻ȱʧֵ
nan_result_pd2 = df.fillna(method='bfill', limit=1)  # �ú����ֵ���ȱʧֵ,����ÿ��ֻ�����һ��ȱʧֵ
nan_result_pd3 = df.fillna(method='pad')  # ��ǰ���ֵ�滻ȱʧֵ
nan_result_pd4 = df.fillna(0)  # ��0�滻ȱʧֵ
nan_result_pd5 = df.fillna({'col2': 1.1, 'col4': 1.2})  # �ò�ֵͬ�滻��ͬ�е�ȱʧֵ
nan_result_pd6 = df.fillna(df.mean()['col2':'col4'])  # ��ƽ��������,ѡ������еľ�ֵ�滻ȱʧֵ
#��ӡ���
print(nan_result_pd1)  # ��ӡ���
print(nan_result_pd2)  # ��ӡ���
print(nan_result_pd3)  # ��ӡ���
print(nan_result_pd4)  # ��ӡ���
print(nan_result_pd5)  # ��ӡ���
print(nan_result_pd6)  # ��ӡ���



















