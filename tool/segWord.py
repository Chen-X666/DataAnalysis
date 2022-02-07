# _*_ coding: utf-8 _*_
"""
Time:     2022/2/7 14:38
Author:   ChenXin
Version:  V 0.1
File:     segWord.py
Describe:  Github link: https://github.com/Chen-X666
"""
import jieba


def getStopwordsList():
    stopwords = [line.strip() for line in open('stopwords.txt',encoding='UTF-8').readlines()]
    return stopwords

def preprocess_text(content_lines, sentences):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
            segs = list(filter(lambda x: x not in getStopwordsList(), segs))  # 去掉停用词
            sentences.append(" ".join(segs))
        except Exception:
            print(line)
            continue
