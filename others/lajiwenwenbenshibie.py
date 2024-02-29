import jieba as jb
# import sklearn
#
# str = "随着计算机技术的不断发展，数字图像在各个领域得到了广泛的应用，如医学影像分析、遥感图像处理、工业检测等。而数字图像分割作为图像处理的基础操作之一，是实现图像识别、目标检测等高级图像处理任务的重要前置步骤。数字图像分割的目标是将图像分为若干个互不重叠的区域，以便对每个区域进行进一步的处理。在实际应用中，数字图像分割的准确性和效率直接影响后续图像处理算法的效果。本文介绍了数字图像分割的基本概念和常用算法，并给出了相应的MATLAB实现。通过对几组典型灰度图像进行分割，我们验证了这些算法的有效性和优缺点，并提出了进一步的改进方案。"
#
# word_list = jb.cut(str)
# # for i in word_list:
# #     print(i,end = '\\')
#
#
# import sklearn.feature_extraction.text as sft
from sklearn.feature_extraction.text import CountVectorizer
#语料库
train_x= ['build fails due publication-tests.xml build target','due to sb']
test_x =['build one to ']
#将文本中的词语转换为词频矩阵  选择前256个词 相当于词向量的维度是256维的
cv_ = CountVectorizer(max_features=256)
#计算个词语出现的次数  此类方法一般先fit拟合，再transform转换
X = cv_.fit_transform(train_x)
#输出语料库
print('corpus',train_x)
#输出词典
print('feature_names',cv_.get_feature_names())
#输出词汇
print('vocabulary_',cv_.vocabulary_)
#输出模型参数
print('params',cv_.get_params(deep=True))
#输出词频
print(X)
#查看词频结果
print(X.toarray())

from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
# from sklearn.naive_bayes import GaussianNB
# corpus = []
# labels = [0,1,0,1,0,1]


#声明两种构建文本特征的模型

# #方法一：基于词频的文本向量
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer1 = CountVectorizer(binary=True)
# #方法2：tfidf方法
# from sklearn.feature_extraction.text import TfidfVectorizer
# vectorizer2=TfidfVectorizer(binary=True)

# X1 = vectorizer1.fit_transform(df.Text)


# 垃圾文本识别
import pandas as pd
df = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\message80W1\\message80W1.csv',encoding = 'utf-8')
df.columns = ['a','b','Text']
df = df.loc[0:100,:]
y = df.b

transformer = TfidfTransformer()
vectorizer = CountVectorizer()
word_vec = vectorizer.fit_transform(df.Text)
words = vectorizer.get_feature_names()
word_cout = word_vec.toarray()
tfidf = transformer.fit_transform(word_cout)
tfidf_ma = tfidf.toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(tfidf_ma,y,test_size=0.20, random_state=100)
print ("训练数据中的样本个数: ", X_train.shape[0], "测试数据中的样本个数: ", X_test.shape[0])

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(tfidf_ma,y)
y_pre = gnb.predict(X_test)

print(type(X_test))
# 任给一个文本，检测是否是垃圾文件
import pandas as pd
df1 = pd.read_csv('D:\\@我的记录文件夹\\临时文件夹\\lajiwenbentest.csv',encoding = 'ANSI')
df1.columns = ['a','b','Text']
transformer = TfidfTransformer()
vectorizer = CountVectorizer()
word_vec = vectorizer.fit_transform(df1.Text)
words = vectorizer.get_feature_names()
word_cout = word_vec.toarray()
tfidf = transformer.fit_transform(word_cout)
tfidf_ma = tfidf.toarray()

gnb.predict(tfidf_ma)