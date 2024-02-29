import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\city1.csv')

#假如我要构造一个聚类数为3的聚类器
estimator = KMeans(n_clusters = 3)#构造聚类器
estimator.fit(data.loc[:,['A','B']])#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

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
        plt.scatter(data.loc[i,'A'], data.loc[i, 'B'], color='b')
    elif label_pred[i] == 2:
        plt.scatter(data.loc[i,'A'], data.loc[i, 'B'], color='g')

plt.title("k-means聚类")

plt.show()