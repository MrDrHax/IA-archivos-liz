import theGivens, taylorSeries, potter

def kalman_filter(F, x, S, H, y, Q, R):
    # S es la matriz descompuesta (raíz cuadrada de P) que se actualizará en cada ciclo

    # 1. Actualizar la covarianza usando Givens (predicción de S)
    S = theGivens.givens(F, Q, S)  # S es la raíz cuadrada de P en forma descompuesta

    # 2. Aplicar el algoritmo de Potter para actualizar el estado y S
    x, S = potter.potter(x, S, y, H, R)

    return x, S
