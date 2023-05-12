import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

def plotData(df):
    """
    Function: Plots a time-series dataset as a line chart.
    Parameters:
        df (pandas DataFrame): A DataFrame with time-series data, where each row represents a timestamp and each column represents a variable of interest.
    Returns:
        None
    Outputs:
        - A line chart of the input DataFrame, with the x-axis representing the timestamps and the y-axis representing the values of the variables.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df1 = df [df.index < '2020 MAR']
    df2 =  df [df.index >= '2020 Feb']
    ax.plot_date(df1.index, df1, '-',label="normal data")
    ax.plot_date(df2.index, df2, '-',color='r',label="abnormal data")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    ax.set_xlabel('year')
    ax.set_ylabel('value')
    ax.set_title("GMAK")
    plt.tight_layout()
    # plt.rcParams.update({'font.size':20})
    plt.legend(fontsize=15)
    plt.show()


if __name__ == '__main__':
    data_K550  = pd.read_excel('K550data.xlsx',sheet_name='K55O',parse_dates=['CDID'], squeeze=True,index_col=0)
    data_JVZ8 = pd.read_excel('JVZ8data.xlsx', sheet_name='JVZ8', parse_dates=['CDID'], squeeze=True, index_col=0)
    data_GMAA = pd.read_excel('GMAAdata.xlsx', sheet_name='GMAA', parse_dates=['CDID'], squeeze=True, index_col=0)
    data_GMAK = pd.read_excel('GMAKdata.xlsx', sheet_name='GMAK', parse_dates=['CDID'], squeeze=True, index_col=0)

    plotData(data_GMAK)