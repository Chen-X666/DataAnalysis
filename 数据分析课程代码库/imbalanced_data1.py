#!/usr/bin/env python
#coding:gbk,
import pandas as pd
from imblearn.over_sampling import SMOTE  # �����������SMOTE
from imblearn.under_sampling import RandomUnderSampler  # Ƿ���������RandomUnderSampler
from sklearn.svm import SVC  # SVM�еķ����㷨SVC

# ���������ļ�
df = pd.read_table('data22.txt', sep='\t',
                   names=['col1', 'col2', 'col3', 'col4', 'col5',
                          'label'])  # ��ȡ�����ļ�
x = df.iloc[:, :-1]  # ��Ƭ���õ�����x
y = df.iloc[:, -1]  # ��Ƭ���õ���ǩy
print(df.iloc[0:5,:])
groupby_data_orgianl = df.groupby('label').count()  # ��label���������
print('ԭʼ���ݼ���������ֲ�')
print(groupby_data_orgianl)  

# ʹ��SMOTE�������й���������
model_smote = SMOTE()  # ����SMOTEģ�Ͷ���
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x,y)  # �������ݲ�������������
x_smote_resampled = pd.DataFrame(x_smote_resampled,
                                 columns=['col1', 'col2', 'col3', 'col4',
                                          'col5'])  # ������ת��Ϊ���ݿ���������
y_smote_resampled = pd.DataFrame(y_smote_resampled,
                                 columns=['label'])  # ������ת��Ϊ���ݿ���������
smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],
                            axis=1)  # ���кϲ����ݿ�
groupby_data_smote = smote_resampled.groupby('label').count()  # ��label���������
print(groupby_data_smote)  # ��ӡ�������SMOTE���������ݼ���������ֲ�

# ʹ��RandomUnderSampler��������Ƿ��������
model_RandomUnderSampler = RandomUnderSampler()  # ����RandomUnderSamplerģ�Ͷ���
x_RandomUnderSampler_resampled, y_RandomUnderSampler_resampled = model_RandomUnderSampler.fit_sample(
x,y)  # �������ݲ���Ƿ��������
x_RandomUnderSampler_resampled = pd.DataFrame(x_RandomUnderSampler_resampled,
                                              columns=['col1', 'col2', 'col3',
                                                       'col4',
                                                       'col5'])  # ������ת��Ϊ���ݿ���������
y_RandomUnderSampler_resampled = pd.DataFrame(y_RandomUnderSampler_resampled,
                                              columns=['label'])  # ������ת��Ϊ���ݿ���������
RandomUnderSampler_resampled = pd.concat([x_RandomUnderSampler_resampled, 
y_RandomUnderSampler_resampled],axis=1)  # ���кϲ����ݿ�
groupby_data_RandomUnderSampler = RandomUnderSampler_resampled.groupby(
    'label').count()  # ��label���������
print(groupby_data_RandomUnderSampler)  # ��ӡ�������RandomUnderSampler���������ݼ���������ֲ�

# ʹ��SVM��Ȩ�ص��ڴ�����������
model_svm = SVC(class_weight='balanced')  # ����SVCģ�Ͷ���ָ�����Ȩ��
model_svm.fit(x, y)  # ����x��y��ѵ��ģ��
print(y[0:5])
ynew=model_svm.predict(x)
print(ynew[0:5])
