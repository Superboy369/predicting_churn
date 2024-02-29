from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

boston = load_boston()

X = boston.data
Y = boston.target
X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.2)

from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_tr,Y_tr)

Y_pre = model.predict(X_te)

print(Y_pre)
print(Y_te)

import matplotlib.pyplot as plt
f1 = plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 设置画布支持中文
plt.title("回归模型预测")

plt.plot(np.arange(0,Y_pre.size),Y_pre,linestyle = '-',color = 'r')
plt.plot(np.arange(0,Y_te.size),Y_te,linestyle = '-',color = 'b')

plt.show()

#print(mean_absolute_error(Y_te,Y_pre))