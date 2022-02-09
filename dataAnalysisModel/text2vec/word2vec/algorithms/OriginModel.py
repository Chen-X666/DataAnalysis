# -*- coding:utf-8 -*-
"""
Description: 基于百度百科大语料的word2vec模型

@author: WangLeAi
@date: 2018/9/18
"""
import os
from dataAnalysisModel.text2vec.word2vec.util.DBUtil import DbPoolUtil
from dataAnalysisModel.text2vec.word2vec.util.JiebaUtil import jieba_util
from dataAnalysisModel.text2vec.word2vec.util.PropertiesUtil import prop
from gensim.models import word2vec

CWD = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

class OriginModel(object):
    def __init__(self):
        self.params = prop.get_config_dict(CWD+"/config/w2v.properties")
        self.db_pool_util = DbPoolUtil(db_type="mysql")
        self.train_data_path = CWD+"/gen/ori_train_data.txt"
        self.model_path = CWD+"/model/oriw2v.model"

    @staticmethod
    def text_process(sentence):
        """
        文本预处理
        :param sentence:
        :return:
        """
        # 过滤任意非中文、非英文、非数字
        # regex = re.compile(u'[^\u4e00-\u9fa50-9a-zA-Z\-·]+')
        # sentence = regex.sub('', sentence)
        words = jieba_util.jieba_cut(sentence)
        return words

    def get_train_data(self):
        """
        获取训练数据，此处需要自行修改，最好写入文件而不是直接取到内存中！！！！！
        :return:
        """
        print("创建初始语料训练数据")
        sql = """"""
        sentences = self.db_pool_util.loop_row(origin_model, "text_process", sql)
        with open(self.train_data_path, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(" ".join(sentence) + "\n")

    def train_model(self):
        """
        训练模型
        :return:
        """
        if not os.path.exists(self.train_data_path):
            self.get_train_data()
        print("训练初始模型")
        sentences = word2vec.LineSentence(self.train_data_path)
        model = word2vec.Word2Vec(sentences=sentences, sg=int(self.params["sg"]), vector_size=int(self.params["size"]),
                                  window=int(self.params["window"]), min_count=int(self.params["min_count"]),
                                  alpha=float(self.params["alpha"]), hs=int(self.params["hs"]), workers=6,
                                  epochs=int(self.params["iter"]))
        model.save(self.model_path)
        print("训练初始模型完毕，保存模型")


origin_model = OriginModel()


