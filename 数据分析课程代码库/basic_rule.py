#!/usr/bin/env python
#coding:gbk,
#python��������1
from functools import reduce
a=3
b=a*2
c=a**2#������
print(a,b,c)#��ӡ���

a,b,c=2,3,"ok"#python֧�ֶ��ظ�ֵ
print(a,b,c)
print(type(c))#��ʾc����������

s='I like python'#�ַ�����ֵ
t=s.split(' ')#
print(t)
p=s+' very much'
print(p)
print(p[2:5])
print(p[0:-1])
print(p[2:])
print(p[:-3])

print('c:\windows\nova')
print(r'c:\windows\nova')

if a==1:
	print(a)
else:
	print('a������1')
		
s,k=0,0
while k<101:#0--101����
	k=k+1
	s=s+k
print(s)

s=0#0--100����
for k in range(101):#range���������������У�range(a,b,c)��aΪ���cΪ����Ҳ�����b-1�ĵȲ�����
	s=s+k
print(s)

s=0
if s in range(4):
	print('s��0,1,2,3��')
if s not in range(1,4,1):
	print('s����1,2,3��')

#�б����
a=[1,2,3,4,5,6]
d=[0,0,0]
del a[5]
b=a
c=a[:]
a[0]=999
print(b)
print(c)

a.append(6)
print(a)
a.extend(d)
print(a)

#�б��Ƶ�ʽ
squares=[]
for x in range(10):
	squares.append(x**2)
print(squares)
squares2=[x**2 for x in range(10)]
print(squares2)

#Ԫ�����
tup1=(1,2,"a",[d,2,"c"])
print(tup1)
tup2=tuple(d)
print(tup2)
tuple3=tup1+tup2
print(tuple3)

#�ֵ����
dict1={'list':[1,2,3],1:123,'a':'python','b':(1,2,3)}
for key in dict1:
	print(str(key)+':'+str(dict1[key]))
seq = ('Google', 'Runoob', 'Taobao')
dict2= dict.fromkeys(seq,0)
print("���ֵ�Ϊ : %s" %  str(dict2))

#����
s=set([1,2,2,3])
q={3,4,5}
print(s)
print(s|q)
print(s&q)
print(s-q)
print(s^q)

#python��def�Զ��庯��	
def add2(x):
	return x+2
print (add2(3))

def add2(x=0,y=0):#������Ĭ��ֵ
	return[x+2,y+2]#����һ���б�
print(add2())
print(add2(3,4))

def add3(x,y):
	return x+3,y+3#���ض��ֵ
a,b=add3(1,2)
print(a,b)

#����ʽ���
f=lambda x:x+2#���ں���
g=lambda x,y:x+y
print(f(3),g(1,2))

a=[1,2,3]
b=map(lambda x: x+2,a)#ӳ��
b=list(b)
print(b)

print(reduce(lambda x,y:x*y, range(1,5+1)))#�ݹ�

s=1
for i in range(1,6):
	s=s*i
print(s)

b=filter(lambda x:x>5 and x<8,range(10))#������������ɸѡ����������Ԫ��
print(list(b))



