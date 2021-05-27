import Modified_GSA as mgs
import numpy as np


def qr_alg(A):
    """
    the QR iteration Algorithm implementation, according to section 4.2
    """

    Agag = A.astype('float64')
    n = np.shape(A)[0]
    Qgag = np.identity(n)

    for i in range(n):
        Q, R = mgs.mgs_algorithm(Agag)
        Agag = np.matmul(R, Q)
        Qgag_Q = np.matmul(Qgag, Q)

        if np.logical_and(0.0001 >= (Qgag - Qgag_Q), (Qgag - Qgag_Q) >= -0.0001).all():
            return Agag, Qgag

        Qgag = Qgag_Q

    return Agag, Qgag


def eigengap_heu(M):
    """
    the  Eigengap Heuristic implementation, according to section 4.3
    returns values : k, the matrix of the k eigenvectors of the smallest
    eigenvalues
    """
    Agag, Qgag = qr_alg(M)


    diagonal = np.diagonal(Agag)
    sorted_arr = np.sort(diagonal)
    k = find_k(sorted_arr)

    # sort the matrix Qgag according to the eigenvalues
    i = np.argsort(diagonal)
    Qgag = Qgag[:, i]

    ret_matrix = Qgag[:, :k]  # taking the corresponding k eigenvectors

    return k, ret_matrix


def find_k(arr):
    """
    finding the number k, according to part 4.3
    """
    n = len(arr)

    part1 = arr[:-1].copy()
    part2 = arr[1:].copy()

    delta_arr = part2 - part1
    delta_arr = delta_arr[: n // 2]
    k = np.argmax(delta_arr)

    return k + 1
