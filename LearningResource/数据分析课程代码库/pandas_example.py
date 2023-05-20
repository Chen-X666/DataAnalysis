#!/usr/bin/env python
#coding:gbk,
import pandas as pd
import numpy as np
s1=pd.Series(range(1,20,5))
s2=pd.Series({'����':90,'��ѧ':79})
s3=pd.Series([1,2,3],index=['a','b','c'])
print(s1)
print(s2)
print(s3)
s1[3]=-15
s2['����']=82
print('='*20)

##################ʱ������#########################
print(pd.date_range(start='20190601', end='20190630', freq='5D'))
print(pd.date_range(start='20190601', end='20190630', freq='W'))
print(pd.date_range(start='20190601', periods=5, freq='2D'))
print(pd.date_range(start='20190601', periods=8, freq='3H'))
print(pd.date_range(start='201906010300', periods=12, freq='T'))
print('='*20)

###############################################
df = pd.DataFrame(np.random.randint(1, 20, (5,3)),
index=range(5),columns=['A', 'B', 'C'])
print(df)
print('='*20)
df = pd.DataFrame(np.random.randint(5, 15, (13, 3)),
index=pd.date_range(start='201907150900',end='201907152100',
freq='H'),columns=['��ʳ', '��ױƷ', '����Ʒ'])
print(df)
print('='*20)
df = pd.DataFrame({'����':[87,79,67,92],'��ѧ':[93,89,80,77],
'Ӣ��':[90,80,70,75]},index=['����', '����', '����', '����'])
print(df)
print('='*20)
df = pd.DataFrame({'A':range(5,10), 'B':3})
print(df)
print('='*20)

###############################################
df = pd.read_excel('����Ӫҵ��2.xlsx',
usecols=['����','����','ʱ��','���׶�'])
print(df[:10])
df2 = pd.read_excel('����Ӫҵ��2.xlsx',
skiprows=[1,3,5], index_col=1)
print(df2[:10])

df = pd.read_excel('����Ӫҵ��2.xlsx')
print(df[5:11])
print(df.iloc[5,:])
print(df.iloc[[3,5,10]])
print(df.iloc[[3,5,10],[0,1,4]])
print(df[['����', 'ʱ��', '���׶�']][:5])
print(df[:10][['����', '����', '��̨']])
print(df.loc[[3,5,10], ['����','���׶�']])
print(df[df['���׶�']>1700])
print(df['���׶�'].sum())
print(df[df['ʱ��']=='14��00-21��00']['���׶�'].sum())
print(df[(df.����=='����')&(df.ʱ��=='14��00-21��00')][:10])
print(df[df['��̨']=='����Ʒ']['���׶�'].sum())
print(df[df['����'].isin(['����','����'])]['���׶�'].sum())
print(df[df['���׶�'].between(800,850)])
#######################################################
print('='*20)
print(df.head())
print(df.describe())
print(df['���׶�'].describe())
print(df.nsmallest(3, '���׶�'))
print(df.nlargest(5, '���׶�'))
print(df['����'].max())
index = df['���׶�'].idxmin()
print(index)
print(df.loc[index,'����'])
##########################################
print('='*20)
print(df.sort_values(by=['���׶�','����'], ascending=False)[:12])
print(df.sort_values(by=['���׶�','����'], ascending=[False,True])[:12])

'''d3=pd.read_excel('code3_data.xlsx')#��ȡExcel������Dataframe��ʽ
print(d3.head())
print(d3.describe())

d3data=pd.read_excel('code3_data.xlsx', sheet_name='Sheet1',usecols=[0,1])
print(d3data)

print(d3data['A'])
print(d3data.loc[0])
print(d3data.loc[2][1])
print(d3data.loc[0]['A'])'''

