import numpy as np
from sklearn import preprocessing as pp, datasets
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
import bkmeans

heart = np.genfromtxt('data/sa_heart.csv', delimiter=',', dtype=float, skip_header=True, usecols=(0,1,2,3,4,5,6,7,8))
digits = datasets.load_digits()
wine = datasets.load_wine()

# Preprocessing
X_digits = pp.StandardScaler().fit_transform(digits.data[:200, 1:])
X_heart = pp.StandardScaler().fit_transform(heart[:200, :])
X_wine = pp.StandardScaler().fit_transform(wine.data[:178, :])

tsne = TSNE(n_components=2)

# Cluster original data into labels

# DIGITS
kmeans = KMeans(n_clusters=9).fit(X_digits)
pred1 = kmeans.predict(X_digits)
pred2 = AgglomerativeClustering(n_clusters=9).fit_predict(X_digits)
pred3 = bkmeans.bkmeans(X_digits, 9, 5)

# HEARTS
kmeans = KMeans(n_clusters=2).fit(X_heart)
pred4 = kmeans.predict(X_heart)
pred5 = AgglomerativeClustering(n_clusters=2).fit_predict(X_heart)
pred6 = bkmeans.bkmeans(X_heart, 2, 5)

# WINE
kmeans = KMeans(n_clusters=3).fit(X_wine)
pred7 = kmeans.predict(X_wine)
pred8 = AgglomerativeClustering(n_clusters=3).fit_predict(X_wine)
pred9 = bkmeans.bkmeans(X_wine, 3, 5)

# Apply DR technique
Xn1 = tsne.fit_transform(X_digits)
Xn2 = tsne.fit_transform(X_heart)
Xn3 = tsne.fit_transform(X_wine)


# Compare clustering techniques after DR is applied

color = ['red', 'green', 'blue', 'yellow', 'black', 'orange', 'pink', '#37AB65', '#3DF735',
         '#AD6D70', '#EC2504', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']

fig, ax = plt.subplots(nrows=3, ncols=3)

def scatter(X, y,clusters, plotNr, title, legend=None):
    for n in range(0, clusters):
        ax = plt.subplot(3, 3, plotNr)
        filter = X[y == n]
        ax.set_title(title)
        ax.scatter(filter[:, 0], filter[:, 1], color=color[n], marker='o', s=0.5)

# Plot DIGITS
scatter(Xn1, pred1, 9, 1, "KMeans")
scatter(Xn1, pred2, 9, 2, "Hierarchical")
scatter(Xn1, pred3, 9, 3, "Bisect KMeans")

# Plot HEARTS
scatter(Xn2, pred4, 2, 4, "KMeans")
scatter(Xn2, pred5, 2, 5, "Hierarchical")
scatter(Xn2, pred6, 2, 6, "Bisect KMeans")

# Plot WINE
scatter(Xn3, pred7, 3, 7, "KMeans")
scatter(Xn3, pred8, 3, 8, "Hierarchical")
scatter(Xn3, pred9, 3, 9, "Bisect KMeans")

plt.show()





