#coding:gbk,
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # ����TF-IDF�Ĵ�Ƶת������
from sklearn.cluster import KMeans
import jieba.posseg as pseg


# ���ķִ�
def jieba_cut(comment):
    seg_list = pseg.cut(comment)  # ��ȷģʽ�ִ�[Ĭ��ģʽ]
    word_list = [i.word for i in seg_list if i.flag in ['a', 'ag', 'an']] # ֻѡ�����ݴ�׷�ӵ��б���
    return word_list


# ��ȡ�����ļ�
fn = open('comment.txt',encoding='utf8')
comment_list = fn.readlines()  # ��ȡ�ļ�����Ϊ�б�
fn.close()

# word to vector
stop_words = ['��', '��', '��', '��', '��', '+', ' ', '��', '��', '��', '��', '��', '.', '-']  # ����ͣ�ô�
vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=jieba_cut, use_idf=True)  # ����������ģ��
X = vectorizer.fit_transform(comment_list)  # �����۹ؼ����б�ת��Ϊ�������ռ�ģ��

# K��ֵ����
model_kmeans = KMeans(n_clusters=3)  # ��������ģ�Ͷ���
model_kmeans.fit(X)  # ѵ��ģ��

# ����������
cluster_labels = model_kmeans.labels_  # �����ǩ���
print(model_kmeans.labels_)
word_vectors = vectorizer.get_feature_names()  # ������
print(word_vectors)

word_values = X.toarray()  # ����ֵ
print(word_values )
comment_matrix = np.hstack(
(word_values, cluster_labels.reshape(word_values.shape[0], 1)))  # ������ֵ�ͱ�ǩֵ�ϲ�Ϊ�µľ���
word_vectors.append('cluster_labels')  # ���µľ����ǩ�б�׷�ӵ�����������
comment_pd = pd.DataFrame(comment_matrix, columns=word_vectors)  # ���������������;����ǩ�����ݿ�
print(comment_pd.head(3))  # ��ӡ������ݿ��1������

# ����������
comment_cluster1 = comment_pd[comment_pd['cluster_labels'] == 2].drop(
'cluster_labels',axis=1)  # ѡ������ǩֵΪ2�����ݣ���ɾ�����һ��
word_importance = np.sum(comment_cluster1, axis=0)  # ���մ�����������ͳ��
print(word_importance.sort_values(ascending=False)[:5])  # ������ͳ�Ƶ�ֵ���������򲢴�ӡ���ǰ5����
