from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
Y = iris.target
X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.2)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_tr,Y_tr)
Y_pre = model.predict(X_te)
model.score(X_te,Y_te)