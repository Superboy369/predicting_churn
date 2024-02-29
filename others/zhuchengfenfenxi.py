import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

digits = load_digits()

mdata = digits.data

print(np.shape(mdata))

pca = PCA(n_components=10)

pca.fit(mdata)

print(pca.explained_variance_ratio_)
print(pca.explained_variance_)

# X_new = pca.transform(mdata)
# plt.scatter(mdata[:, 0],mdata[:, 1],marker='o')
# plt.show()

plt.figure()
for i in range(0,np.shape(mdata)[0]):
    for j in range(0,np.shape(mdata)[1]):
        plt.scatter(i,j)

plt.show()