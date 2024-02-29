from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

digits = load_digits()

X = digits.data
Y = digits.target
X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size = 0.2)

model = GaussianNB()
model.fit(X_tr,Y_tr)
Y_pre = model.predict(X_te)

print(Y_te,"\n",Y_pre)
print(accuracy_score(Y_te,Y_pre))