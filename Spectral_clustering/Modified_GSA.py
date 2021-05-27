import numpy as np
import math

"""
this file implements the modified gram schmidt algorithm, according to section 4.1 at 
the project file. 
"""


def my_euclid(vec):
    """
    compute the euclidian norm
    """
    s = np.square(vec).sum(axis=0)
    return math.sqrt(s)


def mgs_algorithm(matrix):
    """
    the body of the algorithm.
    """
    U = matrix.astype('float64').copy()
    n = np.shape(U)[0]

    R = np.zeros((n, n))
    Q = np.zeros((n, n))

    for i in range(n):
        ext_loop(U, R, Q, i, n)

    return Q, R


def ext_loop(U, R, Q, i, n):
    """
    the body of the external for loop.
    """
    Ui = U[:, i]
    R[i, i] = my_euclid(Ui)
    Qi = Ui / R[i, i]
    Q[:, i] = Qi

    for j in range(i + 1, n):
        in_loop(U, R, Qi, i, j)


def in_loop(U, R, Qi, i, j):
    """
    implementation of the 2 row of code inside the internal for loop"
    """
    R[i, j] = np.matmul(Qi, U[:, j])
    U[:, j] = U[:, j] - R[i, j] * Qi
