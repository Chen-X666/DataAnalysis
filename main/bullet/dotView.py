# _*_ coding: utf-8 _*_
"""
Time:     2021/10/28 15:52
Author:   ChenXin
Version:  V 0.1
File:     dotView.py
Describe:  Github link: https://github.com/Chen-X666
"""
import pydot as pydot
import pydotplus
import os

from matplotlib.pyplot import clf
from sklearn import tree
import pydot

f = open('tree.dot')
print(f.read())
graph = pydot.graph_from_dot_data(f.read())
print(graph)
#graph.write_pdf('iris.pdf')
graph[0].write('iris.pdf')