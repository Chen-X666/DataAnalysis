# _*_ coding: utf-8 _*_
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve


def ROCComparison(models,y_test):


    # Loop that creates the plot. I will pass each ROC curve one by one.
    for m in models:
        auc = roc_auc_score(y_true=y_test,
                            y_score=m['probs'])
        fpr, tpr, thresholds = roc_curve(y_test,
                                         m['probs'])
        plt.plot(fpr, tpr, label='%s ROC (area = %0.3f)' % (m['label'], auc))

    # Settings
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    # Plot!
    plt.show()

def ConfusionMatrixComparison():
    print()

def EvaluatedValueComparison():
    print()