
#coding:gbk,
#ʹ��read��readline��readlines��ȡ����

fn=open('text.txt')
print('���λ�ã�'+str(fn.tell()))
data1=fn.read()
print('�������ݣ�\n'+data1)
print('���λ�ã�'+str(fn.tell()))
line1=fn.readline()
print('��һ�����ݣ�\n'+line1)
fn.close

fn=open('text.txt')
print('���λ�ã�'+str(fn.tell()))
line1=fn.readline()
print('��һ�����ݣ�\n'+line1)
line2=fn.readline()
print('�ڶ������ݣ�\n'+line2+'\n')

fn.close
fn=open('text.txt','r')
line3=fn.readlines()
print('�������ݣ�')
print(line3)

fn=open('text.txt','a+')
fn.write('\n')
fn.write('line3')
