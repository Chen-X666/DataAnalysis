# _*_ coding: utf-8 _*_
"""
Time:     2021/7/24 13:12
Author:   ChenXin
Version:  V 0.1
File:     WordPretreatment.py
Describe:  Github link: https://github.com/Chen-X666
"""
import jieba
import jieba.analyse as anls  # 关键词提取

def getStopwords(text):
    stopwords = [line.strip() for line in open('stopwords.txt', encoding='UTF-8').readlines()]  # list类型
    print(stopwords)
    text_split = jieba.cut(text)  # 未去掉停用词的分词结果   list类型
    text_split_no = []
    for word in text_split:
        if word not in stopwords:
            text_split_no.append(word)

def keywordByTfidf(text_split_no_str):
    keywords = []
    for x, w in anls.extract_tags(text_split_no_str, topK=100, withWeight=True):
        keywords.append(x)  # 前20关键词组成的list
    keywords = ' '.join(keywords)  # 转为str
    print(keywords)
