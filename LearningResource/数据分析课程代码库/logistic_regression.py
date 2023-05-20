#coding:gbk,
#�߼��ع�
import pandas as pd
import numpy as np  # numpy��
from sklearn.linear_model import LinearRegression, LogisticRegression  # ��������Ҫʵ�ֵĻع��㷨

#������ʼ��
filename = 'bankloan.xls'
data = pd.read_excel(filename)
x = data.iloc[:,:-1]
y = data.iloc[:,-1]


model_1 =LinearRegression()  # ������ͨ���Իع�ģ�Ͷ���
model_lr=LogisticRegression(max_iter=1000)
model_dic = [model_1, model_lr]  # ��ͬ�ع�ģ�Ͷ���ļ���
test=x.iloc[3,:].values  #��dataframeת��array

for model in model_dic: 
    model.fit(x,y) 
    print((model.score(x,y)))  
    print(model.predict(np.array(test).reshape(1,-1)))
    

    

