# _*_ coding: utf-8 _*_
"""
Time:     2022/2/7 11:50
Author:   ChenXin
Version:  V 0.1
File:     text2vec.py
Describe:  Github link: https://github.com/Chen-X666
"""
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer


def tfIDF(sentences):
    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=1)
    # 统计每个词语的tf-idf权值
    transformer = TfidfTransformer()
    # 第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
    tfidf = transformer.fit_transform(vectorizer.fit_transform(sentences))
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    weight = tfidf.toarray()
    # 查看特征大小
    print('Features length: ' + str(len(word)))
    return weight

def word2vec():
    print()