import numpy as np

def fitness(sol, matriz_ordenes, matriz_pasillos):
    '''
    Entradas:
    -------------------------------------------------------------------
    sol(list): n+k+2 entradas
        - 1ra entrada: numero n (cantidad de ordenes por recolectar)
        - n's entradas sig: j_1,...,j_n (orden numero j_s por recolectar)
        - n+2 entrada: numero k (cantidad de pasillos visitados)
        - últimas k entradas: i_1,...,i_k (pasillo numero i_s por visitar)
        (formato del pdf)
    matriz_ordenes, matriz_pasillos: formato que le dieron ya caro y bryan
    Salida:
    --------------------------------------------------------------------
    fit (int): puntuacion a la solucion
    
    '''
    # Extraer componentes de la solución
    n = sol[0]  # Número de órdenes
    ordenes = sol[1:n+1]  # Índices de órdenes
    m = sol[n+1]  # Número de pasillos visitados
    pasillos_visit = sol[n+2:n+2+m]  # Índices de pasillos

    # Calcular unidades totales en las órdenes seleccionadas
    unidades_totales = sum(sum(matriz_ordenes[i]) for i in ordenes)

     # Evitar división por cero
    fit = unidades_totales / m if m > 0 else 0
    
    return fit
