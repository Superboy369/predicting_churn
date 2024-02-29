import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from display import plot    # 导入自定义的绘图模块

airline_data = pd.read_csv('D:\\@我的记录文件夹\\生产实习python深度学习\\air_data.csv', encoding='gb18030')  # 以指定格式读取数据
print('原始数据的尺寸为：',airline_data.shape)

'''
丢弃票价为空的记录
'''
exp1 = airline_data['SUM_YR_1'].notnull()
exp2 = airline_data['SUM_YR_2'].notnull()
airline_notnull = airline_data[exp1 & exp2]
print('删除缺失记录后数据的尺寸为：',airline_notnull.shape)

'''
丢弃票价为0，平均折扣率不为0，总飞行公里数大于0的记录
'''
index1 = airline_notnull['SUM_YR_1'] == 0
index2 = airline_notnull['SUM_YR_1'] == 0
index3 = airline_notnull['avg_discount'] != 0
index4 = airline_notnull['SEG_KM_SUM'] > 0
airline = airline_notnull[-(index1&index2&index3&index4)]
print('删除异常记录后数据的尺寸为：',airline.shape)

'''
=======================================
部分字段说明：
FFP_DATE：入会时间
LOAD_TIME：观测窗口的结束时间
FLIGHT_COUNT：飞行次数
LAST_TO_END：最后一次乘机时间至观测窗口结束月数
avg_discount：平均折扣系数
SEG_KM_SUM：总飞行公里数
=======================================
'''

'''
构建L指标
'''
airline_selection = airline.loc[:,['FFP_DATE','LOAD_TIME','FLIGHT_COUNT','LAST_TO_END','avg_discount','SEG_KM_SUM']]
L = pd.to_datetime(airline_selection['LOAD_TIME'])-pd.to_datetime(airline_selection['FFP_DATE'])
L = L.astype('str').str.split().str[0]
L = L.astype('int')/30

airline_features = pd.concat((L, airline_selection.iloc[:,2:]),axis=1)   # 合并特征
airline_features.columns = ['temp','FLIGHT_COUNT','LAST_TO_END','avg_discount','SEG_KM_SUM']
# print(airline_features)
airline_features=np.concatenate((airline_features,[airline_features[0]]))
data = StandardScaler().fit_transform(airline_features)                  # 数据标准化

# data = np.concatenate((data, [data[0]]))

kmeans_model = KMeans(n_clusters=5).fit(data)   # K-means聚类分析

plot(kmeans_model, airline_features.columns)    # 绘制客户分群结果
