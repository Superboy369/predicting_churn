import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('D:\\@我的记录文件夹\\生产实习python深度学习\\06.数据\\data\\susceptibility.csv',encoding = 'utf-8')

data1 = pd.get_dummies(data.loc[:,['geologic structure','human activity','underground water']],prefix = ['col1_','col2_','col3_'])

data1.loc[:,'loc_susceptibility'] = data.loc[:,'susceptibility']

# data1.loc[data1['loc_susceptibility'] == 'likely','loc_susceptibility'] = 1

# data1.loc[data1['loc_susceptibility'] == 'unlikelmscore = model.score(labels_pre,labels_test)y','loc_susceptibility'] = 0

data1 = data1.replace("likely",1)
data2 = data1.replace("unlikely",0)

data = data2.loc[:,['col1__strong', 'col1__weak', 'col2__strong', 'col2__weak', 'col3__poor',
       'col3__rich']]
target = data2.loc[:,'loc_susceptibility']

X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.20, random_state=100)
print ("训练数据中的样本个数: ", X_train.shape[0], "测试数据中的样本个数: ", X_test.shape[0])

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pre = model.predict(X_test)
score = model.score(X_test,y_test)

print(score)

print(y_pre,'\n',y_test)