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
import re

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

def reTest():
    #”[]”中被使用的话就是表示字符类的否定，如果不是的话就是表示限定开头
    # .* 表示任意匹配除换行符
    print('*********** .*表示任意匹配除换行符 ***********')
    line = ['Cas', 'a&c', 'sbmarter', 'sstan', '234697a']
    for i in range(len(line)):
        s = re.match('.a', line[i])
        print(line[i])
        print(s)
    print('************* 比如 ^A会匹配"An e"中的A，但是不会匹配"ab A"中的A**********')
    for i in range(len(line)):
        s1 = re.match(r'^ss', line[i])
        print(line[i])
        print(s1)

    print('2*************')
    for i in range(len(line)):
        s1 = re.match(r'a+', line[i])

        print(s1)
    print('3*************')
    for i in range(len(line)):
        s1 = re.match(r'Ca*', line[i])
        print(s1)
    print('4*************')
    for i in range(len(line)):
        s1 = re.match(r's[s|b]m', line[i])
        print(s1)
    print('5*************数字')
    for i in range(len(line)):
        s1 = re.match(r'\d', line[i])
        print(s1)
    print('6*************字母')
    for i in range(len(line)):
        s1 = re.match(r'\D', line[i])
        print(s1)
    print('7*************')
    for i in range(len(line)):
        s1 = re.match(r'a\Wc', line[i])
        print(s1)

    tt = "Tina is a gOod girl, she is cool, clever, and so on..."
    rr = re.compile(r'\woo\w', re.I)
    print(rr.findall(tt))

    print(re.match('www', 'www.runoob.com').span())  # 在起始位置匹配
    print(re.match('com', 'www.runoob.com'))  # 不在起始位置匹配
    print('***********')
    print(re.search('www', 'www.runoob.com').span())  # 在起始位置匹配
    print(re.search('com', 'www.runoob.com').span())  # 不在起始位置匹配

    p = re.compile(r'\d+')
    print(p.findall('o1n2m3k4'))

    print(re.split('\d+', 'one1two2three3four4five5'))
    print(re.split('\W+', 'runoob, runoob, runoob.'))
    print(re.split('(\W+)', 'runoob, runoob, runoob.'))
    print(re.split('a+', 'hello world'))

    text = 'python is a kind of computer language, very useful...'
    print(re.sub(r'\s+', '-', text))


if __name__ == '__main__':
    reTest()

