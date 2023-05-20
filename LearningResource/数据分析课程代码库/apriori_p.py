#!/usr/bin/env python
#coding:gbk,
#��������

import sys
sys.path.append('./apri')
import pandas as pd
import apriori

# ���������ļ�
fileName = 'menu_orders.xls'
#fileName='menu_orders.xls'
# ͨ�������Զ����apriori����������
minS = 0.1  # ������С֧�ֶȷ�ֵ
minC = 0.1  # ������С���Ŷȷ�ֵ
dataSet = apriori.createData(fileName)  # ��ȡ��ʽ�������ݼ�
print(dataSet)
L, suppData = apriori.apriori(dataSet, minSupport=minS)  # ����õ�������С֧�ֶȵĹ���
rules = apriori.generateRules(fileName, L, suppData, minConf=minC)  # ����������С���ŶȵĹ���
# ���������������
model_summary = 'data record: {1} \nassociation rules count: {0}'  # չʾ���ݼ���¼�������㷧ֵ����Ĺ�������
print (model_summary.format(len(rules), len(dataSet)))  # ʹ��str.format����ʽ�����
df = pd.DataFrame(rules, columns=['item1', 'item2', 'instance', 'support', 'confidence', 'lift'])  # ����Ƶ���������ݿ�
df_lift = df[df['lift'] > 1.0]  # ֻѡ��������>1�Ĺ���
print(df_lift.sort_values('instance', ascending=False))  # ��ӡ���������ݿ�



