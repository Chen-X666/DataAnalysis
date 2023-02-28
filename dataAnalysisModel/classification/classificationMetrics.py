import numpy as np # numpy库
import pandas as pd # pandas库
import prettytable # 图表打印工具
from sklearn import metrics
import matplotlib.pyplot as plt  # 画图工具
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, \
    roc_auc_score  # 分类指标库
import seaborn as sns

sns.set()
# 模型数值评估
def valueEvaluation(y_test,y_predict,y_score):
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # auc
    accuracy = metrics.accuracy_score(y_test, y_predict) # 精度
    confusionmatrix = metrics.confusion_matrix(y_test, y_predict) # 混淆矩阵
    target_names = ['class 0', 'class 1'] # 两个类别
    classifyreport = metrics.classification_report(y_test, y_predict,target_names=target_names) # 分类结果报告
    print('分类准确率 ',accuracy) # 混淆矩阵对角线元素之和/所有元素之和
    print('混淆矩阵 \n', confusionmatrix)
    print('分类结果报告 \n', classifyreport)
    # 核心评估指标：accuracy，precision，recall，f1分数
    accuracy_s = accuracy_score(y_test, y_predict).round(3)  # 准确率
    precision_s = precision_score(y_test, y_predict).round(3)  # 精确度
    recall_s = recall_score(y_test, y_predict).round(3)  # 召回率
    f1_s = f1_score(y_test, y_predict).round(3)  # F1得分
    core_metrics = prettytable.PrettyTable()  # 创建表格实例
    auc_s = auc(fpr, tpr).round(3)  # AUC
    core_metrics.field_names = ['auc','accuracy', 'precision', 'recall', 'f1']  # 定义表格列名
    core_metrics.add_row([auc_s, accuracy_s, precision_s, recall_s, f1_s])  # 增加数据
    print('{:-^60}'.format('核心评估指标'), '\n', core_metrics)


# 混淆矩阵
def ConfusionMatrix(y_test, pre_y):

    TN, FP, FN, TP = confusion_matrix(y_test, pre_y).ravel()  # 获得混淆矩阵并用ravel将四个值拆开赋予TN,FP,FN,TP
    confusion_matrix_table = prettytable.PrettyTable(['', '预测0', '预测1'])
    confusion_matrix_table.add_row(['真实0', TP, FN])
    confusion_matrix_table.add_row(['真实1', FP, TN])
    print('{:-^60}'.format('混淆矩阵'), '\n', confusion_matrix_table)

    # Calculate confusion matrix
    confusion_matrix_rf = confusion_matrix(y_true=y_test,y_pred=pre_y)
    # Turn matrix to percentages
    confusion_matrix_rf = confusion_matrix_rf.astype('float') / confusion_matrix_rf.sum(axis=1)[:, np.newaxis]
    # Turn to dataframe
    df_cm = pd.DataFrame(
        confusion_matrix_rf, index=['0', '1'], columns=['0', '1'],
    )
    # Parameters of the image
    figsize = (10, 7)
    fontsize = 14
    # Create image
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt='.2f')
    # Make it nicer
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0,
                                 ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45,
                                 ha='right', fontsize=fontsize)

    # Add labels
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')  # 子网格标题
    # Plot!
    plt.show()

# ROC曲线
def ROC(modelName,y_test,y_score):
    auc = np.round(roc_auc_score(y_true=y_test,y_score=y_score[:,1]),decimals=3)
    # 核心评估指标：accuracy，precision，recall，f1分数
    fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])  # ROC
    print(fpr,tpr,thresholds)
    # 打印ROC曲线
    #plt.subplot(1, 2, 1)  # 第一个子网格
    plt.plot(fpr, tpr, label=modelName+", auc="+str(auc))  # 画出ROC曲线
    plt.plot([0, 1], [0, 1], linestyle='--', color='k', label='random chance')  # 画出随机状态下的准确率线
    plt.title('ROC')  # 子网格标题
    plt.xlabel('false positive rate')  # X轴标题
    plt.ylabel('true positive rate')  # y轴标题
    plt.legend(loc=4)
    plt.show()  # 展示图形
