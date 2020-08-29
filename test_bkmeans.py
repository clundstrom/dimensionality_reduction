import numpy as np
from sklearn import preprocessing as pp, datasets
import matplotlib.pyplot as plt
import bkmeans


def scatter(X, y, clusters, plotNr, title, legend=None):
    for n in range(0, clusters):
        ax = plt.subplot(1, 1, plotNr)
        filter = X[y == n]
        ax.set_title(title)
        ax.scatter(filter[:, 0], filter[:, 1], color=color[n], marker='o', s=1)


iris = datasets.load_iris()
X_iris = pp.StandardScaler().fit_transform(iris.data)

s1 = np.genfromtxt('data/s1.txt', delimiter=';', dtype=float, skip_header=True)
s1 = pp.StandardScaler().fit_transform(s1)

# Plots
color = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'pink', '#37AB65', '#3DF735',
         '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']

clusters = bkmeans.bkmeans(s1, 15, 15)
scatter(s1, clusters, 15, 1, 'test')
plt.show()