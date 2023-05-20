#!/usr/bin/env python
#coding:gbk,
#������
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score,auc,confusion_matrix,f1_score,precision_score,recall_score,roc_curve#����ָ���
import prettytable
import matplotlib.pyplot as plt

raw_data=np.loadtxt('./data/classification.csv', delimiter=',',skiprows=1)

X=raw_data[:,:-1]
print(type(X))
y=raw_data[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=0)#�����ݼ���Ϊѵ�����Ͳ��Լ�

#ѵ������ģ��
model_tree=tree.DecisionTreeClassifier(random_state=0)#����������ģ��
model_tree.fit(X_train,y_train)
pre_y=model_tree.predict(X_test)

n_samples,n_features=X.shape
print('samples: %d \t features: %d' % (n_samples,n_features))
print(70 * '-')

#��������
confusion_m=confusion_matrix(y_test,pre_y)
confusion_matrix_table=prettytable.PrettyTable()
confusion_matrix_table.add_row(confusion_m[0,:])
confusion_matrix_table.add_row(confusion_m[1,:])
print('confusion matrix')
print(confusion_matrix_table)

# ��������ָ��
y_score = model_tree.predict_proba(X_test)  # ��þ�������Ԥ�����
fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
auc_s = auc(fpr, tpr)  # AUC
accuracy_s = accuracy_score(y_test, pre_y)  # ׼ȷ��
precision_s = precision_score(y_test, pre_y)  # ��ȷ��
recall_s = recall_score(y_test, pre_y)  # �ٻ���
f1_s = f1_score(y_test, pre_y)  # F1�÷�
core_metrics = prettytable.PrettyTable()  # �������ʵ��
core_metrics.field_names = ['auc', 'accuracy', 'precision', 'recall', 'f1']  # ����������
core_metrics.add_row([auc_s, accuracy_s, precision_s, recall_s, f1_s])  # ��������
print('core metrics')
print(core_metrics)  # ��ӡ�����������ָ��

# ģ��Ч�����ӻ�
names_list = ['age', 'gender', 'income', 'rfm_score']  # ����ģ��ά���б�
color_list = ['r', 'c', 'b', 'g']  # ��ɫ�б�
plt.figure()  # ��������
# ������1��ROC����
plt.subplot(1, 2, 1)  # ��һ��������
plt.plot(fpr, tpr, label='ROC')  # ����ROC����
plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')  # �������״̬�µ�׼ȷ����
plt.title('ROC')  # ���������
plt.xlabel('false positive rate')  # X�����
plt.ylabel('true positive rate')  # y�����
plt.legend(loc=0)
# ������2��ָ����Ҫ��
feature_importance = model_tree.feature_importances_  # ���ָ����Ҫ��
print(feature_importance)
plt.subplot(1, 2, 2)  # �ڶ���������
plt.bar(np.arange(feature_importance.shape[0]), feature_importance, tick_label=names_list, color=color_list)  # ��������ͼ

plt.title('feature importance')  # ���������
plt.xlabel('features')  # x�����
plt.ylabel('importance')  # y�����
plt.suptitle('classification result')  # ͼ���ܱ���
plt.show()  # չʾͼ��


# ģ��Ӧ��
X_new = [[40, 0, 55616, 0], [17, 0, 55568, 0], [55, 1, 55932, 1]]
print('classification prediction')
for i, data in enumerate(X_new):
    y_pre_new = model_tree.predict(np.array(data).reshape(1, -1))
    print('classification for %d record is: %d' % (i + 1, y_pre_new))

