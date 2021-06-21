#!/usr/bin/env python
#coding:gbk,

import random  # �����׼��
import numpy as np  # �����������

def simpleSampling(data):
    print('1�����������')
    data = np.loadtxt('data3.txt')  # ������ͨ�����ļ�
    print(np.shape(data))
    data_sample = random.sample(data,2000)
    print(data_sample[:2])  # ��ӡ���ǰ2������
    print(len(data_sample))  # ��ӡ�������������
    return data_sample

def systematicSampling(data):
    print('\n2���Ⱦ����Ҳ��ϵͳ����')
    sample_count = 2000  # ָ����������
    record_count = data.shape[0]  # ��ȡ���������
    print(np.size(data,0))
    width = record_count / sample_count  # ����������
    data_sample = []  # ��ʼ���հ��б�������ų����������
    i = 0  # ���������Եõ���Ӧ����ֵ
    while len(data_sample) <= sample_count and i * width <= record_count - 1:  # ��������С�ڵ���ָ�������������Ҿ�����������Ч��Χ��ʱ
        data_sample.append(data[int(i * width)])  # ��������
        i += 1  # ������
    print(data_sample[:2])  # ��ӡ���ǰ2������
    print(len(data_sample))  # ��ӡ�����������
    return data_sample
def stratifiedSampling(data):
    print('3���ֲ����')
    # �����б�ǩ�������ļ�
    data2 = np.loadtxt('data2.txt')  # ������зֲ��߼�������
    print(np.shape(data2))
    each_sample_count = 200  # ����ÿ���ֲ�ĳ�������
    label_data_unique = np.unique(data2[:, -1])  # �����һ�е����ݶ���ֲ�ֵ��
    print(label_data_unique)
    sample_data = []  # ������б����ڴ�����ճ�������
    sample_dict = {}  # ������ֵ䣬������ʾ���ֲ���������
    for label_data in label_data_unique:  # ����ÿ���ֲ��ǩ
        sample_list = []  # ������б����ڴ����ʱ�ֲ�����
        for data_tmp in data2:  # ��ȡÿ������
            if data_tmp[-1] == label_data:  # ����������һ�е��ڱ�ǩ
                sample_list.append(data_tmp)  #
        print(len(sample_list))
        each_sample_data = random.sample(sample_list,each_sample_count)  # ��ÿ�����ݶ��������
        sample_data.extend(each_sample_data)  # ����������׷�ӵ�����������
        sample_dict[label_data] = len(each_sample_data)  # ������ͳ�ƽ��
    print(sample_dict)  # ��ӡ���������ͳ�ƽ��

def ClusterSampling(data):
    print('4����Ⱥ����')
    data3 = np.loadtxt('data4.txt')  # �����Ѿ����ֺ���Ⱥ�����ݼ�
    print(np.shape(data3))
    label_data_unique = np.unique(data3[:, -1])  # ������Ⱥ��ǩֵ��
    print(label_data_unique)  # ��ӡ���������Ⱥ��ǩ
    sample_label = random.sample(set(label_data_unique), 2)  # �����ȡ2����Ⱥ
    sample_data = []  # ������б������洢���ճ�������
    for each_label in sample_label:  # ����ÿ����Ⱥ��ǩֵ��
        for data_tmp in data3:  # ����ÿ������
            if data_tmp[-1] == each_label:  # �ж������Ƿ����ڳ�����Ⱥ
                sample_data.append(data_tmp)  # ������ӵ����ճ������ݼ�
    print(sample_label)  # ��ӡ���������Ⱥ��ǩ
    print(len(sample_data))  # ��ӡ����ܳ������ݼ�¼����
    return sample_data

if __name__ == '__main__':
    data = []
