import numpy as np
from sklearn.cluster import KMeans


def bkmeans(X, k_clusters, iter):
    clusters = []
    clusters = findClusters(X, k_clusters, iter, clusters)
    merged = mergeClusters(clusters)
    sorted = sorter(X, merged)
    return sorted[:, 2]


def sorter(A, B):
    """
    Function by Rafael Martins

    Uses three sorts. Two to align A and B. And one sort to fit B back into the order of A.

    """
    index = np.arange(A.shape[0])[A[:, 0].argsort()].argsort()
    return B[B[:, 0].argsort()][index]


def mergeClusters(clusters):
    # Pop first cluster the initial group
    merged = np.array(clusters.pop(0)[0])
    merged = np.c_[merged, np.zeros(merged.shape[0])]

    # Merge the rest
    for idx, cluster in enumerate(clusters):
        cluster = np.c_[cluster[0], np.zeros(cluster[0].shape[0])]
        cluster[:, 2] = idx + 1  # set index of cluster
        merged = np.append(merged, cluster, axis=0)
    return merged


def nextSplit(saved_clusters):
    largest_sse = 0

    for idx, cluster in enumerate(saved_clusters):
        # X values of indexes,

        sse = sum(sum((cluster[0] - cluster[1]) ** 2))
        if sse > largest_sse:
            largest_sse = sse
            splitIdx = idx

    return saved_clusters.pop(splitIdx)[0]


def findClusters(X, k_clusters, iter, saved_clusters):
    if len(X) / 2 < 2:
        raise Exception("Cannot split. Reduce nr of clusters for data set.")

    # Split into two clusters
    clf = KMeans(n_clusters=2, n_init=iter)

    # Predictions
    pred = clf.fit_predict(X)

    # Add clusters to list
    saved_clusters.append((X[pred == 0], clf.cluster_centers_[0]))
    saved_clusters.append((X[pred == 1], clf.cluster_centers_[1]))

    # Exit condition
    if len(saved_clusters) == k_clusters:
        return saved_clusters

    next = nextSplit(saved_clusters)

    return findClusters(next, k_clusters, iter, saved_clusters)
