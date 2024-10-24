import numpy as np

def taylorSeries(A, delta_t, order=2):
    F = np.eye(A.shape[0])
    current_term = np.eye(A.shape[0])
    factorial = 1

    for n in range(1, order + 1):
        current_term = np.dot(current_term, A) * delta_t / n
        F += current_term

    return F
