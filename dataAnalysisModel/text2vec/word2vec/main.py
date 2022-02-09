# -*- coding:utf-8 -*-
"""
Description:

@author: WangLeAi
@date: 2018/9/18
"""
import os
from algorithms.Word2VecModel import w2v_model


def main():
    root_path = os.path.split(os.path.realpath(__file__))[0]
    if not os.path.exists(root_path + "/model"):
        os.mkdir(root_path + "/model")
    w2v_model.load_model()
    print(w2v_model.get_sentence_vector("不知不觉间我已经忘记了爱"))


if __name__ == "__main__":
    main()
