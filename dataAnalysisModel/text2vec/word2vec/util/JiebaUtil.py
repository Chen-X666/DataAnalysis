# -*- coding:utf-8 -*-
"""
Description: jieba分词工具类。需要预先下载字典!

@author: WangLeAi
@date: 2018/9/18
"""
import os

import jieba
import jieba.posseg as pseg

CWD = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

class JiebaUtil(object):
    def __init__(self):
        jieba.set_dictionary(CWD+"/gen/dict.txt")
        # 拆分停止词
        with open(CWD+"/gen/stop.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()
            stop_list = [i.strip('\r\n') for i in lines]
        self.stop_set = set(stop_list)
        self.word_type = ('an', 'n', 'Ng', 'nt', 'nz', 'un', 'vg', 'v', 'vn')

    def jieba_cut(self, sentence):
        """
        分词
        :param sentence:
        :return:
        """
        seg_list = list(jieba.cut(sentence, cut_all=False))
        rst = []
        for w in seg_list:
            if w not in self.stop_set:
                rst.append(w)
        return rst

    def jieba_cut_flag(self, sentence):
        """
        含词性分词
        :param sentence:
        :return:
        """
        seg_list = pseg.cut(sentence)
        rst = []
        for w in seg_list:
            if (w.word not in self.stop_set) & (len(w.word) > 1) & (w.flag in self.word_type):
                rst.append(w.word)
        return rst


jieba_util = JiebaUtil()
