import loadData, LDLT, taylorSeries, kalmanFilter
import numpy as np
import matplotlib.pyplot as plt

data = loadData.loadFromFile()

# plt.plot(data)
# plt.show()

# Inicializar el estado y la matriz de covarianza
x = data[0]  # Estado inicial (14 sensores)
P = np.cov(...)  # Matriz de covarianza inicial
L, D = LDLT.LdlT(P)  # Realizar la factorización LDLT de P una sola vez
S = L @ np.sqrt(D)  # Obtener S a partir de LDLT

# Pre-calcular F usando Series de Taylor
A = ...  # La matriz A del sistema NO EXISTE
delta_t = 1 / 128
F = taylorSeries.taylorSeries(A, delta_t)

# Definir H, Q y R
H = np.eye(14)  # Observa todos los sensores
Q = np.eye(14) * 0.01  # Incertidumbre baja en el proceso CON RESPECTO A Z
# Incertidumbre moderada en las mediciones CON RESPECTO A W
R = np.eye(14) * 0.05

output = []

# Ciclo del filtro de Kalman
for x_raw in data:
    # Observación actual

    # Llamar al filtro de Kalman con Givens y Potter
    x, S = kalmanFilter.kalman_filter(F, x, S, H, x_raw, Q, R)

    # Ahora tenemos el estado actualizado x y la nueva descomposición S para el siguiente ciclo
    output.append(x)

