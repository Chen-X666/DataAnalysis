#!/usr/bin/env python
#coding:gbk,

import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing

# ��ȡ����
df = pd.read_table('data7.txt', names=['id', 'amount', 'income', 'datetime','age'])  # ��ȡ�����ļ�
print(df.head(5))  # ��ӡ���ǰ5������

print('���ʱ�����ݵ���ɢ��')
for i, signle_data in enumerate(df['datetime']):  # ѭ���õ������Ͷ�Ӧֵ
    single_data_tmp = pd.to_datetime(signle_data)  # ��ʱ��ת��Ϊdatetime��ʽ
    df['datetime'][i] = single_data_tmp.weekday()  # ��ɢ��Ϊ�ܼ�
print(df.head(5))  # ��ӡ���ǰ5������

print('��Զ�ֵ��ɢ���ݵ���ɢ��')
map_df = pd.DataFrame(
    [['0-10', '0-40'], ['10-20', '0-40'], ['20-30', '0-40'], ['30-40', '0-40'],
     ['40-50', '40-80'],
     ['50-60', '40-80'], ['60-70', '40-80'], ['70-80', '40-80'],
     ['80-90', '>80'], ['>90', '>80']],
    columns=['age', 'new_age'])  # ����һ��Ҫת����������
print(map_df)
df_tmp = df.merge(map_df,how='inner')  # ���ݿ����ƥ��
df = df_tmp.drop('age', 1)  # ������Ϊage����
print(df.head(5))  # ��ӡ���ǰ5������

'''
df1=pd.DataFrame({'key':['a','b','j'],'data1':range(3)})    
df2=pd.DataFrame({'key':['a','b','c'],'data2':range(3)})   
print(df1)
print(df2) 
print(pd.merge(df1,df2,how='outer') )
'''

print('����������ݵ���ɢ��')
print('����1���Զ����������ʵ����ɢ��')
bins = [0, 200, 1000, 5000, 10000]  # �Զ�������߽�
df['amount1'] = pd.cut(df['amount'], bins)  # ʹ�ñ߽�����ɢ��
print(df.head(5))  # ��ӡ���ǰ5������

print('����2 ʹ�þ��෨ʵ����ɢ��')
data = df['amount']  # ��ȡҪ��������ݣ���Ϊamount����
data_reshape = data.values.reshape((data.shape[0], 1))  # ת��������״
model_kmeans = KMeans(n_clusters=4, random_state=0)  # ����KMeansģ�Ͳ�ָ��Ҫ��������
keames_result = model_kmeans.fit_predict(data_reshape)  # ��ģ����
print(type(keames_result))
df['amount2'] = keames_result  # ����ɢ�������ݺϲ���ԭ���ݿ�
print(df.head(5))  # ��ӡ���ǰ5������

print('����3��ʹ��4��λ��ʵ����ɢ��')
df['amount3'] = pd.qcut(df['amount'], 4, labels=['bad', 'medium', 'good','awesome'])  # ���ķ�λ�����зָ�
df = df.drop('amount', 1)  # ������Ϊamount����
print(df.head(5))  # ��ӡ���ǰ5������

print('����������ݵĶ�ֵ��')
binarizer_scaler = preprocessing.Binarizer(threshold=df['income'].mean())  # ����Binarizerģ�Ͷ���
#����ָ������ֵ��������ֵ����С�ڵ�����ֵ�ģ�������ֵ����0����������ֵ�ĸ���1������ֵthresholdĬ�϶�Ϊ0
income_tmp = binarizer_scaler.fit_transform(df[['income']])  # Binarizer��׼��ת��
df['income'] = income_tmp  # Binarizer��׼��ת��
print(df.head(5))  # ��ӡ���ǰ5������

