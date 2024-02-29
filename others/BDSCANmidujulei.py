import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

data = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\city1.csv')

#DBSCAN聚类
db = DBSCAN(eps=2, min_samples=1)
db.fit(data.loc[:,['A','B']])
label_pred = db.labels_
print(label_pred)
#
# for i in range(0,len(label_pred)):
#     print(data.loc[i,'城市'],label_pred[i])

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 设置画布支持中文
plt.subplot(2,2,2)

plt.figure()
for i in range(0,len(label_pred)):
    if label_pred[i] == 0:
        plt.scatter(data.loc[i,'A'],data.loc[i,'B'],color = 'r')
    elif label_pred[i] == 1:
        plt.scatter(data.loc[i,'A'], data.loc[i,'B'], color='b')
    elif label_pred[i] == 2:
        plt.scatter(data.loc[i,'A'], data.loc[i,'B'], color='g')

plt.title("BDSCAN聚类")

plt.show()