# -*- coding: utf-8 -*-

# 导入库
import numpy as np  # 导入numpy库
import pandas as pd  # 导入pandas库
from sklearn.ensemble import GradientBoostingRegressor  # 集成方法回归库
from sklearn.model_selection import GridSearchCV  # 导入网格搜索交叉检验库
import matplotlib.pyplot as plt  # 导入图形展示库

# 读取数据
raw_data = pd.read_table('data/products_sales.txt', sep=',')

# 数据审查和校验
# 数据概览
print('{:*^60}'.format('Data overview:'))
print(raw_data.tail(2))  # 打印原始数据后2条   head(5)
print('{:*^60}'.format('Data dtypes:'))
print(raw_data.dtypes)  # 打印数据类型
print('{:*^60}'.format('Data DESC:'))
print(raw_data.describe().round(1).T)  # 打印原始数据基本描述性信息

# 查看值域分布
col_names = ['limit_infor', 'campaign_type', 'campaign_level', 'product_level']  # 定义要查看的列
for col_name in col_names:  # 循环读取每个列
    unique_value = np.sort(raw_data[col_name].unique())  # 获得列唯一值
    print('{:*^50}'.format('{1} unique values:{0}').format(unique_value, col_name))  # 打印输出

# 缺失值审查
na_cols = raw_data.isnull().any(axis=0)  # 查看每一列是否具有缺失值
print('{:*^60}'.format('NA Cols:'))
print(na_cols)  # 查看具有缺失值的列
print('Total NA lines is: {0}'.format(raw_data.isnull().any(axis=1).sum()))  # 查看具有缺失值的行总记录数

# 相关性分析
print('{:*^60}'.format('Correlation Analyze:'))
short_name = ['li', 'ct', 'cl', 'pl', 'ra', 'er', 'price', 'dr', 'hr', 'cf', 'orders']
long_name = raw_data.columns
name_dict = dict(zip(long_name, short_name))#组成字典
print(name_dict)
print(raw_data.corr().round(2).rename(index=name_dict, columns=name_dict))  # 输出所有输入特征变量以及预测变量的相关性矩阵


# 数据预处理
# 异常值处理
sales_data = raw_data.fillna(raw_data['price'].mean())  # 缺失值替换为均值
sales_data = sales_data.drop('email_rate',axis=1) # 丢弃高相关属性
sales_data = sales_data[sales_data['limit_infor'].isin([0, 1])]  # 只保留促销值为0和1的记录
sales_data['campaign_fee'] = sales_data['campaign_fee'].replace(33380, sales_data[
    'campaign_fee'].mean())  # 将异常极大值替换为均值
print('{:*^60}'.format('transformed data:'))
print(sales_data.describe().round(2).T.rename(index=name_dict))  # 打印处理完成数据基本描述性信息

# 分割数据集X和y
X = sales_data.iloc[:, :-1]  # 分割X
y = sales_data.iloc[:, -1]  # 分割y

# 模型最优化参数训练及检验
model_gbr = GradientBoostingRegressor()  # 建立GradientBoostingRegressor回归对象
parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
              'learning_rate': [0.05, 0.1, 0.15],
              'min_samples_split': [2, 3],
              'min_samples_leaf': [1, 2, 4]}  # 定义要优化的参数信息
model_gs = GridSearchCV(estimator=model_gbr, param_grid=parameters,cv=5, n_jobs=-1)  # 建立交叉检验模型对象，并行数与CPU一致
model_gs.fit(X, y)  # 训练交叉检验模型
print('Best score is:', model_gs.best_score_)  # 获得交叉检验模型得出的最优得分
print('Best parameter is:', model_gs.best_params_)  # 获得交叉检验模型得出的最优参数


# 获取最佳训练模型
model_best = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象
print(model_best.score(X,y))
plt.style.use("ggplot")  # 应用ggplot自带样式库
plt.figure()  # 建立画布对象
plt.plot(np.arange(X.shape[0]), y, linestyle='-', color='k', label='true y')  # 画出原始变量的曲线
plt.plot(np.arange(X.shape[0]), model_best.predict(X), linestyle=':', color='m',
         label='predicted y')  # 画出预测变量曲线
plt.legend(loc=0)  # 设置图例位置
plt.show()  # 展示图像

