import numpy as np


def get_itter(x_p: np.ndarray, S_p: np.ndarray, y: np.ndarray, H: np.ndarray, R: np.ndarray):
    '''
    x -> A vector
    S -> A matrix
    '''

    x: np.ndarray = x_p
    S = S_p
    n = len(y)

    for i in range(n):
        H_i: np.ndarray = H[i]
        y_i: float = y[i]
        R_i: float = R[i, i]
        phi_i: np.ndarray = S.T @ H_i.T
        a_i = 1/((phi_i.T @ phi_i) + R_i)
        gamma_i = a_i / (1 + np.sqrt(a_i * R_i))
        S = S * (np.identity(n)-(a_i*gamma_i * (phi_i@phi_i.T)))

        K_i: np.ndarray = np.dot(S, phi_i)
        x = x + np.dot(K_i, (y_i - H_i @ x))

    return x, S


def potter(x_p_t, S_p_t, y_t, H, R):
    n = y_t.shape[0]
    S = S_p_t.astype(float)
    x = x_p_t.astype(float)
    phi = np.zeros(H.shape[1])
    K = np.zeros(H.shape[1])
    I = np.eye(S.shape[0])

    for i in range(n):
        H_i = H[i, :]
        y_i = y_t[i]
        R_i = R[i, i]

        phi = np.dot(S.T, H_i.T)
        alpha = 1 / (np.dot(phi.T, phi) + R_i)
        gamma = alpha / (1 + np.sqrt(alpha * R_i))
        S = S * (I - (alpha * gamma * np.dot(phi, phi.T)))
        K = np.dot(S, phi)
        x += np.dot(K, y_i - np.dot(H_i, x))

    return x, S



if __name__ == "__main__":
    print(get_itter(
        x_p=np.array([0, 0]),
        S_p=np.array([[1, 0],
                      [0, 1]]),
        y=np.array([1, 2]),
        H=np.array([[1, 0],
                    [0, 1]]),
        R=np.array([[0.1, 0],
                    [0, 0.1]])
    ))
