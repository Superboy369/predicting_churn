from sklearn.datasets import load_linnerud
from sklearn.model_selection import train_test_split
from sklearn import svm

digits = load_linnerud() # 导入数据集

X = digits.data
Y = digits.target
X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.2) # 数据集分成两部分

model = svm.SVC(kernel = 'linear',C=1) # 机器学习模型

model.fit(X_tr,Y_tr) # 将一部分数据导入模型
Y_pre = model.predict(X_te) # 根据另一部分数据预测
model.score(X_te,Y_te)

print(Y_te,Y_pre)