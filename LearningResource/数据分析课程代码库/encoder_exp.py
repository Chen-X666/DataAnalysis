#coding:gbk,
# ����/one-hot����
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le=LabelEncoder()#��ǩ����
le.fit([1,5,6,100])
print(le.transform([1,1,100,6,5]))
#����ɢ������ת����0��n-1֮�������n�ǲ�ͬȡֵ�ĸ���
le.fit(['��','��'])
re=le.transform(['��','��'])
print(re)


ohe=OneHotEncoder()#���ȱ���  sparse=True  ���һ��matrixϡ�����
ohe.fit([[1],[2],[3],[4],[5],[6],[7]])
re=ohe.transform([[2],[4],[1],[4]]).toarray()
print(re)

ohe=OneHotEncoder()#���ȱ���
ohe.fit([['Z'],['z']])
re=ohe.transform([['z'],['Z']]).toarray()
print(re)


