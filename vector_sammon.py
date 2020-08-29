import numpy as np
from scipy.spatial.distance import pdist, squareform


def sammon(X, iter, threshold, alpha):
    np.random.seed(10)
    y = np.random.rand(X.shape[0], 2)

    # X pairwise distances
    dist_x = squareform(pdist(X))
    dist_x_sum = np.sum(dist_x)
    c = np.divide(-2, dist_x_sum)

    for it in range(0, iter):
        # Y pairwise distances
        dist_y = squareform(pdist(y))

        # Stress
        E = ((np.divide(1, dist_x, where=dist_x != 0)) * np.power(np.subtract(dist_x, dist_y), 2)).sum()

        print(f"E({it}): {E}")

        if E < threshold:
            print("Threshold exceeded! Stopping.")
            return y

        #################
        # First partial
        #################

        # For all i (compared to one entire iteration of i), pairwise difference with y[j] ex shape: 600,600,2
        diff_ij = (y[:, None] - y)
        xx, yy = diff_ij[0].shape

        # Filter out all zeroes from all pairwise differences from all y[i]-y[j]
        diff_ij = diff_ij.reshape(xx, xx, 2)

        # Calc all distances at once
        dist_xy = dist_x - dist_y
        mult_xy = np.multiply(dist_y, dist_x)

        # After calculations, flatten the arrays before the sum
        dist_xy = dist_xy.reshape(xx * xx, 1)
        mult_xy = mult_xy.reshape(xx * xx, 1)

        # No division by zero!
        mult_xy[np.where(mult_xy < 1e-9)] = 1e-9

        # Flatten the diff matrix
        diff_ij = diff_ij.reshape(xx * xx, 2)

        # Sum is now ALL divisions for All i and All j
        summ0 = np.multiply(np.divide(dist_xy, mult_xy, where=mult_xy != 0), diff_ij)

        # Reshape sum again
        summ0 = summ0.reshape(xx, xx, 2)

        # Calculate partial of i
        partial0 = c * np.sum(summ0, axis=0)
        partial0 = -1 * partial0
        #################
        # Second partial
        #################

        # Calc all of part 0
        p0 = np.divide(1, mult_xy, where=mult_xy != 0)

        #  Differences between yi and yj  divided
        dist_y1 = dist_y.reshape(xx * xx, 1)
        p1 = np.divide(diff_ij ** 2, dist_y1, where=dist_y1 != 0)
        p2 = 1 + np.divide(dist_xy, dist_y1, where=dist_y1 != 0)

        calc = (p0 * (dist_xy - p1 * p2))

        summ1 = np.sum(calc.reshape(xx, xx, 2), axis=0)
        partial1 = np.abs(np.divide(np.multiply(-2, summ1), dist_x_sum))
        delta = np.divide(partial0, partial1, where=partial1 != 0)

        y = y - alpha * delta
    return y