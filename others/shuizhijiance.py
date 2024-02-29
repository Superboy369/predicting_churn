from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from 水质检测 import *

data,labels = get_img_data() # 获取数据及预处理

# print(data,labels)

data_tr,data_te,labels_tr,labels_te = train_test_split(data,labels,test_size = 0.2)
Dtc = DecisionTreeClassifier().fit(data_tr,labels_tr)  # 模型训练
pre = Dtc.predict(data_te)   # 模型预测

exp = sum(pre == labels_te) / len(pre)  # 预测精度
confusion_matrix = (labels_te,pre)      # 混淆矩阵
classification_report(labels_te,pre)    # 分类属性报告

print(exp)
print(confusion_matrix)