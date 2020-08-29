import numpy as np
from sklearn.metrics import pairwise_distances


def sammon(X, iter, threshold, alpha):
    np.random.seed(10)
    y = np.random.rand(X.shape[0], 2)

    dist_x = pairwise_distances(X, X, 'euclidean')
    dist_x_sum = np.sum(dist_x)
    counter = 0
    previous = -1
    c = np.divide(-2, dist_x_sum)

    for it in range(0, iter):

        # Calc distances
        dist_y = pairwise_distances(y, y, 'euclidean')

        # Calc stress
        E = ((np.divide(1, dist_x, where=dist_x != 0)) * np.power(np.subtract(dist_x, dist_y), 2)).sum()
        print(f"E({it + 1}): {E}")

        if E < threshold:
            print("Threshold exceeded! Stopping.")
            return y

        if E < previous:
            counter = 0
        else:
            counter += 1

        if counter > 3:  # Three consecutive increases in stress
            print("Convergence exploded. Returning current result..")
            return y
        previous = E

        for i in range(0, X.shape[0]):
            # Create a full array of y[i] to calc all y[i] - y[j] at the same time in an attempt to speed up process
            diff = np.full((y.shape), y[i]) - y

            # Initialise sums
            summ0 = np.array([0, 0])
            summ1 = np.array([0, 0])

            for j in range(i + 1, X.shape[0]):

                # First partial
                numerator = dist_x[i, j] - dist_y[i, j]
                denominator = np.multiply(dist_x[i, j], dist_y[i, j])

                if denominator < 1e-9:  # clamp denominator
                    denominator = 1e-9

                summ0 = summ0 + np.multiply(np.divide(numerator, denominator), diff[j])

                # Second partial
                p0 = np.divide(1, denominator)
                p1 = np.divide(np.power(diff[j], 2), dist_y[i, j])
                p2 = 1 + np.divide(numerator, dist_y[i, j])
                summ1 = summ1 + (p0 * (numerator - p1 * p2))

            partial0 = (c * summ0)
            partial1 = np.abs(c * summ1)

            delta = np.divide(partial0, partial1, where=partial1 != 0)
            y[i] -= alpha * delta
    return y
