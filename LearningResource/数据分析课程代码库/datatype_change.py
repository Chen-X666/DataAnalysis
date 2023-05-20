# -*- coding: GBK -*-
# ���������ݺ�˳������ת��Ϊ��־����

import pandas as pd  # ����pandas��
from sklearn.preprocessing import OneHotEncoder  # ����OneHotEncoder��

# ��������
df = pd.DataFrame({'id': [3566841, 6541227, 3512441],
                   'gender': ['male', 'Female', 'Female'],
                   'level': ['high', 'low', 'middle'],
                   'e':[1,2,3]})
print(df)  # ��ӡ���ԭʼ���ݿ�

# �Զ���ת��������
df_new = df.copy()  # ����һ���µ����ݿ������洢ת�����

for col_num, col_name in enumerate(df):  # ѭ������ÿ���е�����ֵ������
#enumerate() �������ڽ�һ���ɱ��������ݶ���(���б�Ԫ����ַ���)���Ϊһ���������У�ͬʱ�г����ݺ������±꣬һ������ for ѭ�����С�
    col_data = df[col_name]  # ���ÿ������
    col_dtype = col_data.dtype  # ���ÿ��dtype����
    print(col_data.dtype)
    if col_dtype == 'object':  # ���dtype������object������ֵ�ͣ���ִ������
        df_new = df_new.drop(col_name,axis=1)  # ɾ��df���ݿ���Ҫ���б�־ת������
        value_sets = col_data.unique()  # ��ȡ�����˳�������Ψһֵ��
        for value_unique in value_sets:  # ��ȡ�����˳������е�ÿ��ֵ
            col_name_new = col_name + '_' + value_unique  # �����µ�������ʹ��ԭ����+ֵ�ķ�ʽ����
            col_tmp = df.iloc[:, col_num]  # ��ȡԭʼ������
            #print(col_tmp)
            new_col = (col_tmp == value_unique)  # ��ԭʼ��������ÿ��ֵ���бȽϣ���ͬΪTrue������ΪFalse
            df_new[col_name_new] = new_col  # Ϊ���ս������������ֵ
print(df_new)  # ��ӡ���ת��������ݿ�

print('\nʹ��sklearn���б�־ת��')
df2 = pd.DataFrame({'id': [3566841, 6541227, 3512441],
                    'gender': [1, 2, 2],
                    'level': [3, 1, 2]})
print(df2)
id_data = df2.values[:, :1]  # ���ID��
transform_data = df2.values[:, 1:]  # ָ��Ҫת������

enc = OneHotEncoder()  # ����ģ�Ͷ���
df2_new = enc.fit_transform(transform_data).toarray()  # ��־ת��
df2_all = pd.concat((pd.DataFrame(id_data), pd.DataFrame(df2_new)),
                    axis=1)  # ���Ϊ���ݿ�
print(df2_all)  # ��ӡ���ת��������ݿ�

