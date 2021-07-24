# _*_ coding: utf-8 _*_
"""
Time:     2021/7/24 13:19
Author:   ChenXin
Version:  V 0.1
File:     dataWordCloud.py
Describe:  Github link: https://github.com/Chen-X666
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt  # 绘制图像的模块

def wordCloud(keywords):
    # 画词云
    wordcloud = WordCloud(
        # 设置字体，不然会出现口字乱码，文字的路径是电脑的字体一般路径，可以换成别的
        font_path="simhei.ttf",
        # 设置了背景，宽高
        background_color="white", width=1000, height=880,
        # 频率最大单词字体大小
        max_font_size=300).generate(keywords)  # keywords为字符串类型

    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()