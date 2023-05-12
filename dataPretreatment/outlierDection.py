# _*_ coding: utf-8 _*_
"""
Time:     2021/6/26 12:07
Author:   ChenXin
Version:  V 0.1
File:     outlierDection.py
Describe:  Github link: https://github.com/Chen-X666
"""
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import IsolationForest
import pandas as pd
from sklearn.svm import OneClassSVM # 导入OneClassSVM
import numpy as np # 导入numpy库
import matplotlib.pyplot as plt # 导入Matplotlib

def SVM_outlier(df):
    raw_data = df.values
    train_set = raw_data[:900, :]  # 训练集
    test_set = raw_data[:100, :]  # 测试集
    # 异常数据检测
    model_onecalsssvm = OneClassSVM(nu=0.1, kernel="rbf")  # 创建异常检测算法模型对象
    model_onecalsssvm.fit(train_set)  # 训练模型
    pre_test_outliers = model_onecalsssvm.predict(test_set)  # 异常检测
    # 异常结果统计
    toal_test_data = np.hstack((test_set, pre_test_outliers.reshape(test_set.shape[0], 1)))  # 将测试集和检测结果合并
    normal_test_data = toal_test_data[toal_test_data[:, -1] == 1]  # 获得异常检测结果中集
    outlier_test_data = toal_test_data[toal_test_data[:, -1] == -1]  # 获得异常检测结果异常数据
    n_test_outliers = outlier_test_data.shape[1]  # 获得异常的结果数量
    total_count_test = toal_test_data.shape[0]  # 获得测试集样本量
    print('outliers: {0}/{1}'.format(n_test_outliers, total_count_test))  # 输出异常的结果数量
    print('{:*^60}'.format(' all result data (limit 5) '))  # 打印标题
    print(toal_test_data[:5])  # 打印输出前5条合并后的数据集
    # 异常检测结果展示
    plt.style.use('ggplot')  # 使用ggplot样式库
    fig = plt.figure()  # 创建画布对象
    ax = Axes3D(fig)  # 将画布转换为3D类型
    s1 = ax.scatter(normal_test_data[:, 0], normal_test_data[:, 1], normal_test_data[:, 2], s=100, edgecolors='k',
                    c='g',
                    marker='o')  # 画出正常样本点
    s2 = ax.scatter(outlier_test_data[:, 0], outlier_test_data[:, 1], outlier_test_data[:, 2], s=100, edgecolors='k',
                    c='r',
                    marker='o')  # 画出异常样本点
    ax.w_xaxis.set_ticklabels([])  # 隐藏x轴标签，只保留刻度线
    ax.w_yaxis.set_ticklabels([])  # 隐藏y轴标签，只保留刻度线
    ax.w_zaxis.set_ticklabels([])  # 隐藏z轴标签，只保留刻度线
    ax.legend([s1, s2], ['normal points', 'outliers'], loc=0)  # 设置两类样本点的图例
    plt.title('novelty detection')  # 设置图像标题
    plt.show()

#异常值处理，通过箱型图去将1.5倍1/4到3/4长度的值去掉
def boxing_outlier(df, columns):
    for col in columns:
        s=df[col]
        oneQuoter=s.quantile(0.25)
        threeQuote=s.quantile(0.75)
        irq=threeQuote-oneQuoter
        min=oneQuoter-1.5*irq
        max=threeQuote+1.5*irq
        df=df[df[col]<=max]
        df=df[df[col]>=min]
    return df


def ZScore_outlier(df, columns,threshold=3):
    for col in columns:
        mean_df = df[col].mean()
        std_df = df[col].std()
        df['z_score'] = (df[col] - mean_df) / std_df
        df = df[df['z_score'] >= threshold]
        del df['z_score']
    return df

def isolationForest(data):
    # 创建模型，n_estimators：int，可选（默认值= 100），集合中的基本估计量的数量
    model_isof = IsolationForest(n_estimators=20)
    # 计算有无异常的标签分布
    outlier_label = model_isof.fit_predict(data)
    # 将array 类型的标签数据转成 DataFrame
    outlier_pd = pd.DataFrame(outlier_label, columns=['outlier_label'])

    # 将标签数据与原来的数据合并
    data_merge = pd.concat((data, outlier_pd), axis=1)

    print(data_merge['outlier_label'].value_counts())


    # 取出异常样本
    outlier_source = data_merge[data_merge['outlier_label'] == -1]
    print(outlier_source)

    # 取出正常样本
    normal_source = data_merge[data_merge['outlier_label'] == 1]
    print(normal_source)
    data_merge = data_merge.drop(data_merge[data_merge['outlier_label']==-1].index)
    del data_merge['outlier_label']
    return data_merge


from pycaret.anomaly import *
# 分解日期特征
def create_features(df):
    df['year'] = df.index.year #年
    df['month'] = df.index.month #月
    df['dayofmonth'] = df.index.day #日
    df['dayofweek'] = df.index.dayofweek #星期
    df['quarter'] = df.index.quarter #季度
    df['weekend'] = df.dayofweek.apply(lambda x: 1 if x > 5 else 0) #是否周末
    df['dayofyear'] = df.index.dayofyear   #年中第几天
    df['weekofyear'] = df.index.weekofyear #年中第几月
    df['is_month_start']=df.index.is_month_start
    df['is_month_end']=df.index.is_month_end
    return df

def PyOD(df):
    # 创建特征
    df4 = create_features(df.copy())
    # 异常值算法：'knn','cluster','iforest','svm'等。
    alg = 'knn'  # 异常值算法
    fraction = 0.02  # 异常值比例 0.02,0.03,0.04,0.05
    # 创建异常值模型
    r = setup(df4.copy(), session_id=123, verbose=False)
    model = r.create_model(alg, fraction=fraction, verbose=False)
    model_results = r.assign_model(model, verbose=False)
    # 获取检测结果
    df5 = pd.merge(df.reset_index(), model_results[['Anomaly']],
                   left_index=True, right_index=True)
    df5.set_index('date', inplace=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    a = df5.loc[df5['Anomaly'] == 1, ['y']]
    ax.plot(df5.index, df5['y'], color='blue', label='正常值')
    ax.scatter(a.index, a['y'], color='red', label='异常值')
    plt.title(f'Pycaret.anomaly {fraction=}')
    plt.xlabel('date')
    plt.ylabel('y')
    plt.legend()
    plt.show()


if __name__ == '__main__':


    rng = np.random.RandomState(42)

    # Generate train data
    X = 0.3 * rng.randn(100, 2)
    X_train = np.r_[X + 2, X - 2]
    # Generate some regular novel observations
    X = 0.3 * rng.randn(20, 2)
    X_test = np.r_[X + 2, X - 2]
    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

    # fit the model
    clf = IsolationForest(max_samples=100, random_state=rng)
    clf.fit(X_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    y_pred_outliers = clf.predict(X_outliers)

    # plot the line, the samples, and the nearest vectors to the plane
    xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.title("IsolationForest")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

    b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                     s=20, edgecolor='k')
    b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                     s=20, edgecolor='k')
    c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                    s=20, edgecolor='k')
    plt.axis('tight')
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.legend([b1, b2, c],
               ["training observations",
                "new regular observations", "new abnormal observations"],
               loc="upper left")
    plt.show()