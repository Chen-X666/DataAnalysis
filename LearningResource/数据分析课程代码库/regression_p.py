#coding:gbk,
# �ع����
import numpy as np  # numpy��
from sklearn.linear_model import BayesianRidge, LinearRegression, ElasticNet  # ��������Ҫʵ�ֵĻع��㷨
from sklearn.svm import SVR  # SVM�еĻع��㷨
from sklearn.ensemble import GradientBoostingRegressor  # �����㷨
from sklearn.model_selection import cross_val_score  # �������
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score  # ��������ָ���㷨
import pandas as pd  # ����pandas
import matplotlib.pyplot as plt  # ����ͼ��չʾ��

# ����׼��
raw_data = np.loadtxt('regression.txt')  # ��ȡ�����ļ�
X = raw_data[:, :-1]  # �ָ��Ա���
y = raw_data[:, -1]  # �ָ������

# ѵ���ع�ģ��
n_folds = 6  # ���ý������Ĵ���
model_br = BayesianRidge()  # ������Ҷ˹��ع�ģ�Ͷ���
model_lr = LinearRegression()  # ������ͨ���Իع�ģ�Ͷ���
model_etc = ElasticNet()  # ������������ع�ģ�Ͷ���
model_svr = SVR()  # ����֧���������ع�ģ�Ͷ���
model_gbr = GradientBoostingRegressor()  # �����ݶ���ǿ�ع�ģ�Ͷ���
model_names = ['BayesianRidge', 'LinearRegression', 'ElasticNet', 'SVR', 'GBR']  # ��ͬģ�͵������б�
model_dic = [model_br, model_lr, model_etc, model_svr, model_gbr]  # ��ͬ�ع�ģ�Ͷ���ļ���

cv_score_list = []  # ����������б�
pre_y_list = []  # �����ع�ģ��Ԥ���yֵ�б�
for model in model_dic:  # ����ÿ���ع�ģ�Ͷ���
    scores = cross_val_score(model, X, y, cv=n_folds,scoring='r2')  # ��ÿ���ع�ģ�͵��뽻�����ģ������ѵ������
    cv_score_list.append(scores)  # ������������������б�
    print('dd')
    print(scores)
    pre_y_list.append(model.fit(X, y).predict(X))  # ���ع�ѵ���еõ���Ԥ��y�����б�


# ģ��Ч��ָ������
n_samples, n_features = X.shape  # ��������,��������
model_metrics_name = [explained_variance_score, mean_absolute_error, mean_squared_error, r2_score]  # �ع�����ָ�����
model_metrics_list = []  # �ع�����ָ���б�
for i in range(5):  # ѭ��ÿ��ģ������
    tmp_list = []  # ÿ����ѭ������ʱ����б�
    for m in model_metrics_name:  # ѭ��ÿ��ָ�����
        tmp_score = m(y, pre_y_list[i])  # ����ÿ���ع�ָ����
        tmp_list.append(tmp_score)  # ���������ÿ����ѭ������ʱ����б�
    model_metrics_list.append(tmp_list)  # ���������ع�����ָ���б�
df1 = pd.DataFrame(cv_score_list, index=model_names)  # ���������������ݿ�
df2 = pd.DataFrame(model_metrics_list, index=model_names, columns=['ev', 'mae', 'mse', 'r2'])  # �����ع�ָ������ݿ�
print ('samples: %d \t features: %d' % (n_samples, n_features))  # ��ӡ�������������������
print (70 * '-')  # ��ӡ�ָ���
print ('cross validation result:')  # ��ӡ�������
print (df1)  # ��ӡ��������������ݿ�
print (70 * '-')  # ��ӡ�ָ���
print ('regression metrics:')  # ��ӡ�������
print (df2)  # ��ӡ����ع�ָ������ݿ�
print (70 * '-')  # ��ӡ�ָ���
print ('short name \t full name')  # ��ӡ�����д��ȫ������
print ('ev \t explained_variance')
print ('mae \t mean_absolute_error')
print ('mse \t mean_squared_error')
print ('r2 \t r2')
print (70 * '-')  # ��ӡ�ָ���

# ģ��Ч�����ӻ�
plt.figure()  # ��������
plt.plot(np.arange(X.shape[0]), y, color='k', label='true y')  # ����ԭʼֵ������
color_list = ['r', 'b', 'g', 'y', 'c']  # ��ɫ�б�
linestyle_list = ['-', '.', 'o', 'v', '*']  # ��ʽ�б�
for i, pre_y in enumerate(pre_y_list):  # ����ͨ���ع�ģ��Ԥ��õ������������
    plt.plot(np.arange(X.shape[0]), pre_y_list[i], color_list[i], label=model_names[i])  # ����ÿ��Ԥ������
plt.title('regression result comparison')  # ����
plt.legend(loc='upper right')  # ͼ��λ��
plt.ylabel('real and predicted value')  # y�����
plt.show()  # չʾͼ��

# ģ��Ӧ��
print('regression prediction')
new_point_set = [[1.05393, 0., 8.14, 0., 0.538, 5.935, 29.3, 4.4986, 4., 307., 21., 386.85, 6.58],
                 [0.7842, 0., 8.14, 0., 0.538, 5.99, 81.7, 4.2579, 4., 307., 21., 386.75, 14.67],
                 [0.80271, 0., 8.14, 0., 0.538, 5.456, 36.6, 3.7965, 4., 307., 21., 288.99, 11.69],
                 [0.7258, 0., 8.14, 0., 0.538, 5.727, 69.5, 3.7965, 4., 307., 21., 390.95, 11.28]]  # ҪԤ��������ݼ�
for i, new_point in enumerate(new_point_set):  # ѭ������ÿ��ҪԤ������ݵ�
    new_pre_y = model_gbr.predict(np.array(new_point).reshape(1, -1))  # ʹ��GBR����Ԥ��
    print('predict for new point %d is:  %.2f' % (i + 1, new_pre_y))  # ��ӡ���ÿ�����ݵ��Ԥ����Ϣ
