# _*_ coding: utf-8 _*_
"""
Time:     2022/12/16 11:57
Author:   ChenXin
Version:  V 0.1
File:     timeSeriesProcess.py
Describe:  Github link: https://github.com/Chen-X666
"""
# 将数据离散化后，要想放入逻辑回归模型中，需要对数据进行处理
# 因为数据中的123是类别不是大小，这个数量关系仅仅表示顺序，他们之间实质性的数值间隔你是不知道的（WOE可以解决这个问题）
# 而我们一般用的方法是哑变量，或独热编码，将特征中的类别提取出来，设为单独的一个特征。
#https://blog.csdn.net/AvenueCyy/article/details/105162470
import scorecardpy as sc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def WoEDistribution(bins):
    plotlist = sc.woebin_plot(bins)
    #save binning plot
    for key,i in plotlist.items():
        i.savefig('pic/'+str(key)+'.png')
        i.show()

# Split in train and test BEFORE we apply WoE
# Use your Student ID as seed!
def WoEBin(train,test,yColumnName):
    #Now we can bin the variables. The function woebin will do this automatically for us.
    # It will use trees sequentially given the constraints we decide. It is good practice to not leave less than 5% of cases in each bin, and I am using 50 starting bins. It might be a bit less given the data is small (say, 20), but it is not terribly important at this stage.
    #Tip: For larger datasets, use a relatively large number of bins (50 to 100), for smaller ones, use less.3v
    bins = sc.woebin(train, y = yColumnName,
                     min_perc_fine_bin=0.02, # How many bins to cut initially into
                     min_perc_coarse_bin=0.05,  # Minimum percentage per final bin
                     stop_limit=0.1, # Minimum information value
                     max_num_bin=8, # Maximum number of bins
                     method='tree')

    #plotlist = sc.woebin_plot(bins)
    # save binning plot
    # for key,i in plotlist.items():
    #     i.show()
        # plt.savefig(str(key)+'.png')
    return train, test, bins

def adjustWoEByAuto(train, bins,yColumnName):
    breaks_adj = sc.woebin_adj(train, yColumnName, bins)
    #in python 3.7 and in woebin_adj
    # plt.show(woebin_plot(binx)[x_i])->woebin_plot(binx)[x_i].show()
    bins_adj = sc.woebin(train, y=yColumnName, breaks_list=breaks_adj)
    sc.woebin_plot(bins_adj)
    return bins_adj

def adjustWoEByManual(train, breaks_adj,yColumnName):
    bins_adj = sc.woebin(train, y=yColumnName, breaks_list=breaks_adj)
    #plotlist = sc.woebin_plot(bins_adj)
    # for key,i in plotlist.items():
    #     i.show()
    return bins_adj

# 𝐼𝑉<0.02 : No predictive ability, remove.
# 0.02≤𝐼𝑉<0.1 : Small predictive ability, suggest to remove.
# 0.1≤𝐼𝑉<0.3 : Medium predictive ability, leave.
# 0.3≤𝐼𝑉<1 : Good predictive ability, leave.
# 1≤𝐼𝑉 : Strong predictive ability. Suspicious variable.
# Study if error in calculation (i.e. WoE leaves a category with 100% goods or bads) or if variable is capturing future information.
def IVDestribution(train, bins_adj,yColumnName):
    train_woe = sc.woebin_ply(train, bins_adj)  # Calculate WoE dataset (train)
    #train_woe.head()
    IVDestribution = sc.iv(train_woe, yColumnName)
    print(IVDestribution)
    # Check column order.
    print(train_woe.columns)


def IVFiltering(train,test,bins_adj,dropColumns):
    train_woe = sc.woebin_ply(train, bins_adj)  # Calculate WoE dataset (train)
    test_woe = sc.woebin_ply(test, bins_adj)  # Calculate WoE dataset (test)
    # Create range of accepted variables
    train_woe_adj = train_woe.drop(dropColumns, axis=1)
    test_woe_adj = test_woe.drop(dropColumns, axis=1)
    train_woe_adj.head()
    test_woe_adj.head()
    return train_woe_adj,test_woe_adj

if __name__ == '__main__':
    bankloan_data = pd.read_csv('Bankloan.csv')
    bankloan_data = bankloan_data.loc[(bankloan_data['Income'] < 300) &
                                      (bankloan_data['Creddebt'] < 15) &
                                      (bankloan_data['OthDebt'] < 30)]

    bankloan_data.describe()
    X = bankloan_data.iloc[:,1:]
    y = bankloan_data.iloc[:, -1]
    train, test, bins = WoEBin(X=X, y=y)
    #adjustWoEByAuto(train=train,bins=bins)
    breaks_adj = {
        # Below are the intervals for different bins
        'Income': [25.0, 40.0, 55.0, 90.0],
        'OthDebt': [1.0, 2.0, 3.0, 4.0],
    }
    bins_adj = adjustWoEByManual(train=train,breaks_adj=breaks_adj)
    IVDestribution(train=train,test=test,bins_adj=bins_adj)
    train_woe,test_woe = IVFiltering(train=train,test=test,bins_adj=bins_adj,dropColumns=['Education_woe','OthDebt_woe'])
    # Split in train and test BEFORE we apply WoE
    # Use your Student ID as seed!

    train, test = sc.split_df(bankloan_data.iloc[:, 1:],
                              y='Default',
                              ratio=0.7, seed=20190227).values()