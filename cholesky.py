import numpy as np

def cholesky(S, P, n):
    for i in range(n):
        S[i][i] = np.sqrt(P[i][i] - sum([S[j][i] ** 2 for j in range(i)]))
        for j in range(i):
            S[j][i] = (P[j][i]- sum([S[j][k] * S[i][k] for k in range(i)]))/(S[i][i]) if j<i else 0

P = np.array(
    [
        [2,1,0,0],
        [1,2,1,0],
        [0,1,2,1],
        [0,0,1,2],
    ]
    , dtype=np.float64
)

n = 4

S = np.zeros((4, 4), dtype=np.float64)

cholesky(S, P, n)

print(S)
print(S * S.T)
