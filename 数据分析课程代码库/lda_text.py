#coding:gbk,
import os
import pandas as pd
import pyprind
import numpy as np
 
basepath='./data/aclImdb'
labels={'pos':1,'neg':0}


pbar=pyprind.ProgBar(50000)#���ɽ�������50000��ʾ����50000��
df=pd.DataFrame()
for s in ('test','train'):
	for l in ('pos','neg'):
		path=os.path.join(basepath,s,l)#ƴ��·��
		for file in os.listdir(path):
			with open(os.path.join(path,file),'r',encoding='utf-8') as infile:
				txt=infile.read()
			df=df.append([[txt,labels[l]]],ignore_index=True)
			pbar.update()
df.columns=['review','sentiment']
print(df.head())

##��������,���洢ΪCSV����
np.random.seed(0)
df=df.reindex(np.random.permutation(df.index))
df.to_csv('movie_data.csv',index=False,encoding='utf-8')
 
#ת�ɴ�����
df=pd.read_csv('movie_data.csv',encoding='utf-8')
df=df[0:500]

from sklearn.feature_extraction.text import CountVectorizer
count=CountVectorizer(stop_words='english',max_df=.1,max_features=5000)
X=count.fit_transform(df['review'].values)
print(X.shape)

#LDAģ��

from sklearn.decomposition import LatentDirichletAllocation
lda=LatentDirichletAllocation(n_components=10,
                              random_state=1,learning_method='online')
X_topics=lda.fit_transform(X)
print(lda.components_)

#��ʾǰ10��������ÿ�������5������Ҫ�ĵ���
n_top_word=5
feature_names=count.get_feature_names()
for topic_idx, topic in enumerate(lda.components_):
	print('Topic %d:', (topic_idx+1))
	print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_word-1:-1]]))
	


