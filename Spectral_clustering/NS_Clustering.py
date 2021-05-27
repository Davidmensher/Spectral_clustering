from QR_ItAlg import eigengap_heu
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
from kmeans_pp import k_means_pp
from kmeans import kmeans

def compute_W(X):
    """
    compute the Weighted Adjacency Matrix according to section 3.1
    """
    n = np.shape(X)[0]
    W1 = np.zeros((n, n))

    for i in range(n):
        vi = X[i]
        if i < n - 1:
            distances = np.power((X[i + 1:] - vi), 2).sum(axis=1)
            distances = np.sqrt(distances)
            W1[i, i + 1:] += distances
    W2 = np.matrix.transpose(W1)
    W = np.exp((W1 + W2) / -2)
    np.fill_diagonal(W, 0)

    return W


def compute_Lnorm(W):
    """
    compute the Lnorm Matrix according to section 3.1
    """
    n = np.shape(W)[0]

    diagonal = W.sum(axis=1)
    diagonal = np.power(diagonal, -1 / 2)
    D_neg_half = np.zeros((n, n))
    np.fill_diagonal(D_neg_half, diagonal)

    I = np.identity(n)
    part1 = np.matmul(D_neg_half, W)
    part = np.matmul(part1, D_neg_half)

    Lnorm = I - part + I

    return Lnorm 

def compute_T(Lnorm):
    k, U = eigengap_heu(Lnorm)
    T = normalize(U)

    return k, T

def compute_kmeans(T, k, n):
    centroids = k_means_pp(k, n, k, T)
    d, c = kmeans(T.tolist(), centroids.tolist(), k, n, k, 300)

    return d

def assign_points(d, X, T):
    # Re assign points back to original matrix
    clusters = {}
    clusters_idx = []

    for key in d:
        help_arr = []
        for p in d[key]:
            row = np.all(T==p, axis=1)

            for i, r in enumerate(row):
                if r:
                    help_arr.append(i)

        clusters_idx.append(help_arr)

    return clusters_idx

def calculate_ns_clustering(X, n, fixed_k=None):
    W = compute_W(X)
    Lnorm = compute_Lnorm(W)

    k, T = compute_T(Lnorm)

    if fixed_k != None:
        k = fixed_k

    d = compute_kmeans(T, k, n)

    clusters = assign_points(d, X, T)
    return k, clusters