# -*- coding:utf-8 -*-
"""
Description:word2vec fine tuning
基于对应类型的额外语料进行微调

@author: WangLeAi
@date: 2018/9/11
"""
import os
from dataAnalysisModel.text2vec.word2vec.util.DBUtil import DbPoolUtil
from dataAnalysisModel.text2vec.word2vec.util.JiebaUtil import jieba_util
from dataAnalysisModel.text2vec.word2vec.util.PropertiesUtil import prop
from gensim.models import word2vec
from dataAnalysisModel.text2vec.word2vec.algorithms.OriginModel import origin_model


class Word2VecModel(object):
    def __init__(self):
        self.db_pool_util = DbPoolUtil(db_type="mysql")
        self.train_data_path = "gen/train_data.txt"
        self.origin_model_path = "model/oriw2v.model"
        self.model_path = "model/w2v.model"
        self.model = None
        # 未登录词进入需考虑最小词频
        self.min_count = int(prop.get_config_value("config/w2v.properties", "min_count"))

    @staticmethod
    def text_process(sentence):
        """
        文本预处理
        :param sentence:
        :return:
        """
        # 过滤任意非中文、非英文、非数字等
        # regex = re.compile(u'[^\u4e00-\u9fa50-9a-zA-Z\-·]+')
        # sentence = regex.sub('', sentence)
        words = jieba_util.jieba_cut(sentence)
        return words

    def get_train_data(self):
        """
        获取训练数据
        :return:
        """
        print("创建额外语料训练数据")
        sql = """"""
        sentences = self.db_pool_util.loop_row(w2v_model, "text_process", sql)
        with open(self.train_data_path, "a", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(" ".join(sentence) + "\n")

    def train_model(self):
        """
        训练模型
        :return:
        """
        if not os.path.exists(self.origin_model_path):
            print("无初始模型，进行初始模型训练")
            origin_model.train_model()
        model = word2vec.Word2Vec.load(self.origin_model_path)
        print("初始模型加载完毕")
        if not os.path.exists(self.train_data_path):
            self.get_train_data()
        print("额外语料训练")
        extra_sentences = word2vec.LineSentence(self.train_data_path)
        model.build_vocab(extra_sentences, update=True)
        model.train(extra_sentences, total_examples=model.corpus_count, epochs=model.iter)
        model.save(self.model_path)
        print("额外语料训练完毕")

    def load_model(self):
        """
        载入模型
        :return:
        """
        print("载入词嵌入模型")
        if not os.path.exists(self.model_path):
            print("无词嵌入模型，进行训练")
            self.train_model()
        self.model = word2vec.Word2Vec.load(self.model_path)
        print("词嵌入模型加载完毕")

    def get_word_vector(self, words, extra=0):
        """
        获取词语向量，需要先载入模型
        :param words:
        :param extra:是否考虑未登录词，0不考虑，1考虑
        :return:
        """
        if extra:
            if words not in self.model:
                more_sentences = [[words, ] for i in range(self.min_count)]
                self.model.build_vocab(more_sentences, update=True)
                self.model.train(more_sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
                self.model.save(self.model_path)
        rst = None
        if words in self.model:
            rst = self.model[words]
        return rst

    def get_sentence_vector(self, sentence, extra=0):
        """
        获取文本向量，需要先载入模型
        :param sentence:
        :param extra: 是否考虑未登录词，0不考虑，1考虑
        :return:
        """
        words = jieba_util.jieba_cut_flag(sentence)
        if not words:
            words = jieba_util.jieba_cut(sentence)
        if not words:
            print("存在无法切出有效词的句子：" + sentence)
            # raise Exception("存在无法切出有效词的句子：" + sentence)
        if extra:
            for item in words:
                if item not in self.model:
                    more_sentences = [words for i in range(self.min_count)]
                    self.model.build_vocab(more_sentences, update=True)
                    self.model.train(more_sentences, total_examples=self.model.corpus_count, epochs=self.model.iter)
                    self.model.save(self.model_path)
                    break
        return self.get_sentence_embedding(words)

    def get_sentence_embedding(self, words):
        """
        获取短文本向量，仅推荐短文本使用
        句中所有词权重总和求平均获取文本向量，不适用于长文本的原因在于收频繁词影响较大
        长文本推荐使用gensim的doc2vec
        :param words:
        :return:
        """
        count = 0
        vector = None
        for item in words:
            if item in self.model:
                count += 1
                if vector is not None:
                    vector = vector + self.model[item]
                else:
                    vector = self.model[item]
        if vector is not None:
            vector = vector / count
        return vector


w2v_model = Word2VecModel()
