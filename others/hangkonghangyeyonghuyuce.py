import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from display import plot # 导入自定义的绘图模块

airline_data = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\air_data.csv',encoding = 'gb18030')
print('原始数据的尺寸是:',airline_data.shape)

# 数据清洗
# 丢弃票价为空的记录
exp1 = airline_data['SUM_YR_1'].notnull()
exp2 = airline_data['SUM_YR_2'].notnull()
airline_notnull = airline_data(exp1 & exp2)
print('去除确实记录后数据的尺寸是:',airline_data.shape)

# 丢弃
index1 = airline_notnull['SUM_YR_1'] == 0
index2 = airline_notnull['SUM_YR_2'] == 0
index3 = airline_notnull['avg_discount'] != 0
index4 = airline_notnull['SEG_KM_SUM'] > 0
airline = airline_notnull[-(index1 & index2 & index3 & index4)]
print('处理之后尺寸:',airline.shape)