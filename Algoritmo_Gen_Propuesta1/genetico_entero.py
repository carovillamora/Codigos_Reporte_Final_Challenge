import funciones_entero as fn
import numpy as np
import bisect

def inicio(mu, general, ordenes_list, stock, pasillos_list):
    """
    Genera una población inicial de soluciones considerando órdenes completas y penalización por stock.

    Parámetros:
    - mu: número de individuos a generar.
    - general: lista con parámetros globales [ordenes, items, pasillos, wave_lower, wave_upper].
    - ordenes_list: lista de diccionarios {item_id: cantidad} de cada orden.
    - stock: diccionario {item_id: cantidad disponible en stock}.
    - pasillos_list: lista de diccionarios representando los pasillos.

    Retorna:
    - S: lista de soluciones [vector x, pasillos_seleccionados, [sum_i, n_pasillos, fun]].
    """

    S = []

    for _ in range(mu):
        x = np.zeros(general[0], dtype=int)  # Vector binario de órdenes seleccionadas
        sum_i = 0
        sec = np.array(range(general[0]))
        np.random.shuffle(sec)  # Aleatorizar el orden de las órdenes

        # Fase 1: Agregar órdenes completas hasta alcanzar el límite inferior
        i = 0
        while i < len(sec) and sum_i < general[3]:
            orden = ordenes_list[sec[i]]  # Orden completa como diccionario {item_id: cantidad}
            demanda_orden = sum(orden.values())  # Sumar todos los ítems de la orden

            if sum_i + demanda_orden <= general[4]:  # Verificar límite superior
                sum_i += demanda_orden
                i += 1  # Marcar la orden como seleccionada

        # Fase 2: Agregado probabilístico de órdenes completas
        while i < len(sec) and sum_i < general[4] and np.random.uniform() < 0.8:
            orden = ordenes_list[sec[i]]
            demanda_orden = sum(orden.values())

            if sum_i + demanda_orden <= general[4]:
                sum_i += demanda_orden
                i += 1

        x = sec[:i]

        # Generar demanda
        demanda = fn.generar_demanda(ordenes_list, x)

        # Penalización por exceso de stock
        exceso_stock = sum(max(0, cantidad - stock.get(item_id, 0)) for item_id, cantidad in demanda.items())

        if exceso_stock > 0:
            n_pasillos, pasillos_seleccionados = general[2], []
        else:
            n_pasillos, pasillos_seleccionados = fn.pasillos(pasillos_list, demanda, general[2])

        # Calcular función objetivo
        fun = fn.funcion_objetivo(demanda, n_pasillos, general[3], general[4], exceso_stock)

        individuo = [x, pasillos_seleccionados, np.array([sum_i, n_pasillos, fun])]
        S.append(tuple(individuo))

    return S

############ SELECCIÓN ###########

def seleccion(S:np.ndarray, N:int, select:str):
    """
    Realiza la selección de individuos para la siguiente generación.

    Parámetros:
    -----------
    - S: Población de los individuos generados (array de numpy), donde cada individuo incluye:
         - Las primeras n posiciones: variables.
         - La última posición: el valor de la función.
    - select: Esquema de selección

    Retorna:
    --------
    - np.ndarray: Población seleccionada.
    """
    if select == "ruleta":  # Selección por ruleta
        values = np.array([individuo[2][2] for individuo in S]) 
        
        # Manejo de valores negativos (si es necesario)
        if np.any(values <= 0):
            values = values - np.min(values) + 1e-10  # Ajuste para valores positivos
        
        # Cálculo límites acumulativos
        probs = values / np.sum(values)
        limites = np.cumsum(probs)

        # Selección usando ruleta
        M = []
        for _ in range(N):
            r = np.random.uniform(0, 1)
            idx = bisect.bisect_left(limites, r)
            M.append(S[idx])

    elif select == "torneo.rep":  # Torneo binario con reposición
        M = []
        perm = np.random.choice(range(N), N, replace=True)
        for i in range(N):
            # Seleccionar 2 individuos aleatorios
            idx1, idx2 = i, perm[i]
            # Escoger el mejor
            winner = idx1 if S[idx1][2][2] > S[idx2][2][2] else idx2
            M.append(S[winner])

    elif select == "torneo.sin.rep":  # Torneo binario sin reposición
        M = []
        perm = np.random.permutation(N)
        for i in range(N):
            # Seleccionar 2 individuos aleatorios
            idx1, idx2 = i, perm[i]
            # Escoger el mejor
            winner = idx1 if S[idx1][2][2] > S[idx2][2][2] else idx2
            M.append(S[winner])

    return M

############ RECOMBINACIÓN ###########

def recombinacion(M, N, pc, recom):
    """
    Realiza la recombinación únicamente sobre el vector binario de los individuos.
    
    - M: Lista de individuos, donde cada uno tiene [x, pasillos_seleccionados, np.array([sum_i, n_pasillos, f])]
    - N: Número de individuos
    - pc: Probabilidad de cruce
    - recom: Tipo de recombinación ("un.punto" o "dos.puntos")
    
    Retorna:
    - Lista de individuos recombinados con estructura preservada.
    """
    p_prima = []
    ind = 0

    while ind < N:
        if ind + 1 >= len(M):
            p_prima.append(M[ind])
            break       
        
        p1, p2 = M[ind][0], M[ind+1][0]  # Tomamos solo el vector binario `x`
        if len(p1) < 2 or len(p2) < 2:
            nuevo_x1, nuevo_x2 = p1, p2
        elif np.random.uniform() < pc:
            match recom:
                case "un.punto":
                    # Seleccionar un punto de corte distinto para cada padre
                    punto_p1 = np.random.randint(1, len(p1)) 
                    punto_p2 = np.random.randint(1, len(p2))  
                    
                    nuevo_x1 = np.concatenate((p1[:punto_p1], p2[punto_p2:]))
                    nuevo_x2 = np.concatenate((p2[:punto_p2], p1[punto_p1:]))
                    
                case "dos.puntos":
                    # Seleccionar dos puntos de corte distintos por padre
                    puntos_p1 = sorted(np.random.choice(range(1, len(p1)), size=2, replace=False))
                    puntos_p2 = sorted(np.random.choice(range(1, len(p2)), size=2, replace=False))

                    nuevo_x1 = np.concatenate((p1[:puntos_p1[0]], p2[puntos_p2[0]:puntos_p2[1]], p1[puntos_p1[1]:]))
                    nuevo_x2 = np.concatenate((p2[:puntos_p2[0]], p1[puntos_p1[0]:puntos_p1[1]], p2[puntos_p2[1]:]))

        else:
            nuevo_x1, nuevo_x2 = p1, p2  # No hay cruce, se conservan iguales

        # Conservar la estructura original del individuo
        individuo1 = [list(dict.fromkeys(nuevo_x1)), M[ind][1], M[ind][2]]
        individuo2 = [list(dict.fromkeys(nuevo_x2)), M[ind+1][1], M[ind+1][2]]

        p_prima.append(individuo1)
        p_prima.append(individuo2)

        ind += 2  # Procesamos de dos en dos

    return p_prima

def mutacion(p_prima, N, pm, ordenes_list, pasillos_list, general, stock):
    """
    Aplica mutación solo al vector binario de cada individuo y recalcula métricas.
    
    - p_prima: Lista de individuos con la estructura [x, pasillos_seleccionados, np.array([sum_i, n_pasillos, f])]
    - N: Número de individuos a mutar.
    - pm: Probabilidad de mutación por bit.
    - ordenes_list: Lista de diccionarios de órdenes
    - pasillos_list: Lista de diccionarios de pasillos
    - general: Parámetros generales
    - stock: Diccionario de stock disponible
    
    Retorna:
    - Lista de individuos mutados con estructura preservada.
    """
    for i in range(N):
        # Obtener el vector binario del individuo
        x = np.copy(p_prima[i][0])

        for j in range(len(x)):
            if np.random.uniform(0, 1) < pm:
                out = np.random.choice(list(set(range(general[0])) - set(x)))
                x[j] = out

        # Generar demanda consolidada
        demanda = fn.generar_demanda(ordenes_list, x)

        # Penalización por exceso de stock
        exceso_stock = sum(max(0, cantidad - stock.get(item_id, 0)) for item_id, cantidad in demanda.items())

        # En las funciones inicio() y mutacion(), cambiar:
        if exceso_stock > 0:
            n_pasillos, pasillos_seleccionados = general[2], []  # No sumar 100
        else:
            n_pasillos, pasillos_seleccionados = fn.pasillos(pasillos_list, demanda, general[2])

        # Calcular función objetivo (note we removed the stock parameter)
        fun = fn.funcion_objetivo(demanda, n_pasillos, general[3], general[4], exceso_stock)

        # Calcular suma total de demanda
        sum_i = sum(demanda.values())

        # Actualizar el individuo
        p_prima[i] = tuple([x, pasillos_seleccionados, np.array([sum_i, n_pasillos, fun])])

    return p_prima

############ REEMPLAZO ###########

def reemplazo(S, p_prima, N):
    """
    Aplica reemplazo para seleccionar los mejores N individuos basados en la función objetivo `f`.

    - S: Lista de población de padres
    - p_prima: Lista de población de hijos
    - N: Número de individuos a conservar
    
    Retorna:
    - Lista de los mejores N individuos.
    """
    # Combinar población padre e hijo
    poblacion_completa = S + p_prima  
    
    # Extraer valores de la función objetivo `f`
    valores_f = [individuo[2][2] for individuo in poblacion_completa]  
    
    # Ordenar individuos por `f` de MAYOR a menor ([::-1])
    indices_ordenados = np.argsort(valores_f)[::-1]  
    
    # Seleccionar los mejores N individuos
    S = [poblacion_completa[i] for i in indices_ordenados[:N]]
    
    return S
