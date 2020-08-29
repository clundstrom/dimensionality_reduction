import numpy as np
from sklearn import preprocessing as pp, datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import ex2


heart = np.genfromtxt('data/sa_heart.csv', delimiter=',', dtype=float, skip_header=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))
digits = datasets.load_digits()
wine = datasets.load_wine()

X_digits = pp.StandardScaler().fit_transform(digits.data[:200, 1:])
X_heart = pp.StandardScaler().fit_transform(heart[:200, :])
X_wine = pp.StandardScaler().fit_transform(wine.data[:200, :])

y_heart = np.genfromtxt('data/sa_heart.csv', delimiter=',', dtype=float, skip_header=True, usecols=(9))
y_heart = y_heart[:200]
y_heart[y_heart == 1] = 0
y_heart[y_heart == 2] = 1
y_digits = digits.target[:200]
y_wine = wine.target[:200]


pca = PCA(n_components=2)
tsne = TSNE(n_components=2)

# Fit
Xn2 = ex2.sammon1(X_digits.copy(), 5000, 1e-6, 0.02)
Xn1 = pca.fit_transform(X_digits.copy())
Xn3 = tsne.fit_transform(X_digits.copy())

Xn5 = ex2.sammon1(X_heart.copy(), 2500, 1e-6, 0.02)
Xn4 = pca.fit_transform(X_heart.copy())
Xn6 = tsne.fit_transform(X_heart.copy())
#
Xn8 = ex2.sammon1(X_wine.copy(), 3000, 1e-6, 0.02)
Xn7 = pca.fit_transform(X_wine.copy())
Xn9 = tsne.fit_transform(X_wine.copy())


# Plots
color = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'pink', '#37AB65', '#3DF735',
         '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']

fig, ax = plt.subplots(nrows=3, ncols=3)

def scatter(X, y,clusters, plotNr, title, legend=None):
    for n in range(0, clusters):
        ax = plt.subplot(3, 3, plotNr)
        filter = X[y == n]
        ax.set_title(title)
        ax.scatter(filter[:, 0], filter[:, 1], color=color[n], marker='o', s=0.5)

scatter(Xn1, y_digits, 9, 1, "PCA")
scatter(Xn2, y_digits, 9, 2, "Sammon")
scatter(Xn3, y_digits, 9, 3, "t-SNE")

scatter(Xn4, y_heart, 2, 4, "PCA")
scatter(Xn5, y_heart, 2, 5, "Sammon")
scatter(Xn6, y_heart, 2, 6, "t-SNE")
#
scatter(Xn7, y_wine, 3, 7, "PCA")
scatter(Xn8, y_wine, 3, 8, "Sammon")
scatter(Xn9, y_wine, 3, 9, "t-SNE")
plt.show()