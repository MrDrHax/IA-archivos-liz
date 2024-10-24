import numpy as np


def get_U(F: np.ndarray, S: np.ndarray, Q: np.ndarray) -> np.ndarray:
    top = np.dot(F.T, S.T)
    bottom = np.sqrt(Q).T

    return np.vstack((top, bottom))


def calulate_itter(Q_0: np.ndarray, F_0: np.ndarray, S_0: np.ndarray):
    m = S_0.shape[0]
    m2 = m * 2

    U = get_U(F_0, S_0, Q_0)

    for j in range(m):
        for i in range(m2-1, j, -1):
            B = np.identity(m2)

            a = U[i - 1, j]
            b = U[i, j]

            if b == 0:
                c = 1
                s = 0

            else:
                if np.abs(b) > np.abs(a):
                    r = a/b
                    s = 1 / (np.sqrt(1 + r**2))
                    c = s * r
                else:
                    r = b/a
                    c = 1/np.sqrt(1 + r**2)
                    s = c * r

            
            B[ i - 1, i - 1] = c
            B[i - 1, i] = -s
            B[i, i - 1] = s
            B[i, i] = c

            U = np.dot(B.T , U)

    return U[:m, :m]

if __name__ == "__main__":
    print(calulate_itter(
        F_0=np.array([[1,1], [0,1]]),
        S_0=np.array([[1,0],[0,1]]),
        Q_0=np.array([[0,0],[0,2]]),
    ))


def givens(F, Q, S):
    m = S.shape[0]
    U = np.concatenate([np.dot(F.T, S.T), np.sqrt(Q.T)], axis=0)

    for j in range(m):
        for i in range(2 * m - 1, j, -1):
            B = np.eye(2 * m)
            a = U[i-1, j]
            b = U[i, j]

            if b == 0:
                c = 1
                s = 0
            elif abs(b) > abs(a):
                r = a / b
                s = 1 / np.sqrt(1 + r**2)
                c = s * r
            else:
                r = b / a
                c = 1 / np.sqrt(1 + r**2)
                s = c * r

            B[i-1, i-1] = c
            B[i-1, i] = -s
            B[i, i-1] = s
            B[i, i] = c

            U = np.dot(B.T, U)

    S = U[:m, :m]
    return S
