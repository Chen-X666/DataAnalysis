# _*_ coding: utf-8 _*_
"""
Time:     2021/6/26 12:07
Author:   ChenXin
Version:  V 0.1
File:     outlierDection.py
Describe:  Github link: https://github.com/Chen-X666
"""
from sklearn.ensemble import IsolationForest
import pandas as pd


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



if __name__ == '__main__':
    print()
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import IsolationForest

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