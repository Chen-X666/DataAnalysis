#coding:gbk,

# ��Excel��ȡ��Ӫ����
# �����
import xlrd

# ���ļ�
xlsx = xlrd.open_workbook('demo.xlsx')
# �鿴����sheet�б�
print('All sheets: %s' % xlsx.sheet_names())

# �鿴sheet1�����ݸſ�
sheet1 = xlsx.sheets()[0]  # ��õ�һ��sheet��������0��ʼ
sheet1_name = sheet1.name  # �������
sheet1_cols = sheet1.ncols  # �������
sheet1_nrows = sheet1.nrows  # �������

print(
'Sheet1 Name: %s\nSheet1 cols: %d\nSheet1 rows: %d' % (sheet1_name, sheet1_cols, sheet1_nrows))

# �鿴sheet1���ض���Ƭ����
sheet1_nrows4 = sheet1.row_values(4)  # ��õ�5������
sheet1_cols2 = sheet1.col_values(2)  # ��õ�3������
cell23 = sheet1.row(2)[3].value  # �鿴��3�е�4������
cell1=sheet1.row(1)[2].value
cell2=sheet1.row(2)[2].value
print(cell1+cell2)
print('Row 4: %s\nCol 2: %s\nCell 1: %s\n' % (sheet1_nrows4, sheet1_cols2, cell23))

# �鿴sheet1��������ϸ
for i in range(sheet1_nrows):  # ���д�ӡsheet1����
    print(sheet1.row_values(i))
print('\n')
for i in range(sheet1_cols):  # ���д�ӡsheet1����
    print(sheet1.col_values(i))
