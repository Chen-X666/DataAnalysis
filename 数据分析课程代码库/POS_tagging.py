#coding:gbk,
import jieba.posseg as pseg
import jieba.analyse  # ����ؼ�����ȡ��
import pandas as pd
import re
# ��ȡ�ı��ļ�
fn = open('./data/article1.txt')  # ��ֻ����ʽ���ļ�
string_data = fn.read()  # ʹ��read������ȡ�����ı�
fn.close()  # �ر��ļ�����
pattern = re.compile('\t|\n|-|"')  # ����������ʽƥ��ģʽ
string_data = re.sub(pattern, ' ', string_data)
# �ִ�+���Ա�ע
words = pseg.cut(string_data)  # �ִ�
words_list = []  # ���б����ڴ洢�ִʺʹ��Է���
for word in words:  # ѭ���õ�ÿ���ִ�
    words_list.append((word.word, word.flag))  # ���ִʺʹ��Է���׷�ӵ��б�
words_pd = pd.DataFrame(words_list, columns=['word', 'type'])  # ����������ݿ�
print(words_pd.head(5))  # չʾ���ǰ4��

# ���Է������-���з���
words_gb = words_pd.groupby(['type', 'word'])['word'].count()
print(words_gb.head(5))

# ���Է������-���з���
words_gb2 = words_pd.groupby('type').count()
words_gb2 = words_gb2.sort_values(by='word', ascending=False)
print(words_gb2.head(5))

# ѡ���ض����ʹ�����չʾ
words_pd_index = words_pd['type'].isin(['n', 'eng'])
words_pd_select = words_pd[words_pd_index]
print(words_pd_select.head(5))


# �ؼ�����ȡ
print('�ؼ�����ȡ:')
tags_pairs = jieba.analyse.extract_tags(string_data, topK=5, withWeight=True,
                                        allowPOS=['ns', 'n', 'vn', 'v', 'nr'],
                                        withFlag=True)  # ��ȡ�ؼ��ֱ�ǩ
tags_list = []  # ���б������洢��ֺ������ֵ
for i in tags_pairs:  # ��ӡ��ǩ�������TF-IDFȨ��
    tags_list.append((i[0].word, i[0].flag, i[1]))  # ��������ֶ�ֵ
tags_pd = pd.DataFrame(tags_list, columns=['word', 'flag', 'weight'])  # �������ݿ�
print(tags_pd)  # ��ӡ���ݿ�

