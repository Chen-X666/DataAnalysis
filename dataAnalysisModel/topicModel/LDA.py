# _*_ coding: utf-8 _*_
"""
Time:     2022/4/23 11:38
Author:   ChenXin
Version:  V 0.1
File:     LDA.py
Describe:  Github link: https://github.com/Chen-X666
"""
import os
import re
import jieba
import pyLDAvis as pyLDAvis
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import pyLDAvis.sklearn

def top_words_data_frame(model: LatentDirichletAllocation,
                         tf_idf_vectorizer: TfidfVectorizer,
                         n_top_words: int) -> pd.DataFrame:
    '''
    求出每个主题的前 n_top_words 个词

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    tf_idf_vectorizer : sklearn 的 TfidfVectorizer
    n_top_words :前 n_top_words 个主题词

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    rows = []
    feature_names = tf_idf_vectorizer.get_feature_names_out()
    for topic in model.components_:
        top_words = [feature_names[i]
                     for i in topic.argsort()[:-n_top_words - 1:-1]]
        rows.append(top_words)
    columns = [f'topic {i+1}' for i in range(n_top_words)]
    df = pd.DataFrame(rows, columns=columns)

    return df


def predict_to_data_frame(model: LatentDirichletAllocation, X: np.ndarray) -> pd.DataFrame:
    '''
    求出文档主题概率分布情况

    Parameters
    ----------
    model : sklearn 的 LatentDirichletAllocation
    X : 词向量矩阵

    Return
    ------
    DataFrame: 包含主题词分布情况
    '''
    matrix = model.transform(X)
    columns = [f'P(topic {i+1})' for i in range(len(model.components_))]
    df = pd.DataFrame(matrix, columns=columns)
    return df

#数据预处理
def process_data(df):
    document_column_name = '正文'
    pattern = u'[\\s\\d,.<>/?:;\'\"[\\]{}()\\|~!\t"@#$%^&*\\-_=+a-zA-Z，。\n《》、？：；“”‘’｛｝【】（）…￥！—┄－]+'
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.rename(columns={
        document_column_name: 'text'
    })
    # 去重、去缺失、分词
    df['cut'] = (
        df['text']
        .apply(lambda x: str(x))
        .apply(lambda x: re.sub(pattern, ' ', x))
        .apply(lambda x: " ".join(jieba.lcut(x)))
    )
    print(df['cut'])
    return df


if __name__ == '__main__':
    df = pd.read_excel(
        r'C:\Users\Chen\Desktop\dataAnalysisPlatform\dataAnalysisModel\topicModel\2018-2019茂名（含自媒体）.xlsx',
        sheet_name='微信公众号新闻')
    print(df)
    df = process_data(df)
    buildTF_IDF(df)