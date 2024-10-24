import numpy as np

def LdlT(A):
    n = A.shape[0]
    L = np.zeros((n, n))
    D = np.zeros((n, n))

    for i in range(n):
        D[i, i] = A[i, i] - sum(L[i, k] ** 2 * D[k, k] for k in range(i))
        L[i, i] = 1  # Diagonal de L es 1

        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * L[i, k] * D[k, k]
                       for k in range(i))) / D[i, i]

    return L, D
