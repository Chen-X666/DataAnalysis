#coding:gbk,
# �����
import re  # ������ʽ��
import collections  # ��Ƶͳ�ƿ�
import numpy as np  # numpy��
import jieba  # ��ͷִ�
import wordcloud  # ����չʾ��
from PIL import Image  # ͼ�����
import matplotlib.pyplot as plt  # ͼ��չʾ��

# ��ȡ�ı��ļ�
fn = open('./data/article1.txt')  # ��ֻ����ʽ���ļ�
string_data = fn.read()  # ʹ��read������ȡ�����ı�
fn.close()  # �ر��ļ�����

# �ı�Ԥ����
pattern = re.compile('\t|\n|-|:|;|\.|\)|\(|\?|"')  # ����������ʽƥ��ģʽ
string_data = re.sub(pattern, '', string_data)  # ������ģʽ���ַ����滻��
print(string_data)
# �ı��ִ�
seg_list_exact = jieba.cut(string_data, cut_all=False)  # ��ȷģʽ�ִ�[Ĭ��ģʽ]
remove_words = ['��', '��', '��', '��', '����', '����', ' ', '��', '��', '��', '��', '��',
                '��', '��', '��', '��', '��', '��', '����', '����', '��', '��', '��Ҫ', '�ṩ',
                '��', '����', 'ͨ��', '��', '��ͬ', 'һ��', '���', '����', '��', '��',
                'ͬʱ', '��', '���', '��', '��', '�ǳ�', '��', '���', '����', '��']  # �Զ���ȥ���ʿ�

object_list = [i for i in seg_list_exact if i not in remove_words] # ������ȥ�����б��еĴ���ӵ��б���

# ��Ƶͳ��
word_counts = collections.Counter(object_list)  # �Էִ�����Ƶͳ��
word_counts_top5 = word_counts.most_common(5)  # ��ȡǰ10��Ƶ����ߵĴ�
for w, c in word_counts_top5:  # �ֱ����ÿ���ʺͳ��ִӴ���
    print(w, c)  # ��ӡ���

# ��Ƶչʾ
mask = np.array(Image.open('./data/wordcloud.jpg'))  # �����Ƶ����
wc = wordcloud.WordCloud(
    font_path='C:/Windows/Fonts/simhei.ttf',  # ���������ʽ�������ý��޷���ʾ����
    mask=mask,  # ���ñ���ͼ
    max_words=200,  # ���������ʾ�Ĵ���
    max_font_size=100)  # �����������ֵ

wc.generate_from_frequencies(word_counts)  # ���ֵ����ɴ���
image_colors = wordcloud.ImageColorGenerator(mask)  # �ӱ���ͼ������ɫ����
wc.recolor(color_func=image_colors)  # ��������ɫ����Ϊ����ͼ����
plt.imshow(wc)  # ��ʾ����
plt.axis('off')  # �ر�������
plt.show()  # ��ʾͼ��
