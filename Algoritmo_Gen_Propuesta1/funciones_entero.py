import pulp

def lectura(archivo: str):
    """
    Lee el archivo y transforma la información en diccionarios dispersos.

    Retorna:
    - general: lista con parámetros globales [ordenes, items, pasillos, wave_lower, wave_upper]
    - ordenes_dicc: diccionario {orden_id: {item_id: cantidad}}
    - pasillos_dicc: diccionario {pasillo_id: {item_id: cantidad}}
    """

    with open(archivo) as info:
        datos = [list(map(int, linea.split())) for linea in info if linea.strip()]

    # Extraer parámetros generales
    general = datos.pop(0) + datos.pop()

    num_ordenes = general[0]

    ordenes_list = []
    for i, orden in enumerate(datos[:num_ordenes]):
        ordenes = {}
        for j in range(1, len(orden), 2):
            ordenes[orden[j]] = orden[j+1]
        ordenes_list.append(ordenes)

    pasillos_list = []
    for i, pasillo in enumerate(datos[num_ordenes:]):
        pasillos = {}
        for j in range(1, len(pasillo), 2):
            pasillos[pasillo[j]] = pasillo[j+1]
        pasillos_list.append(pasillos)

    return general, ordenes_list, pasillos_list

def generar_demanda(ordenes_list, x):
    """
    Genera un diccionario de demandas a partir de órdenes activadas por un vector binario.

    Parámetros:
    - ordenes_list: lista de diccionarios {item_id: cantidad}, donde cada diccionario representa una orden.
    - x: ordene incomporadas

    Retorna:
    - demanda: diccionario {item_id: cantidad total requerida}
    """

    demanda = {}

    # Iterar sobre las órdenes activas
    for orden_id in x:
        for item_id, cantidad in ordenes_list[orden_id].items():
            demanda[item_id] = demanda.get(item_id, 0) + cantidad  # Acumular cantidades

    return demanda

def generar_stock(pasillos_list):
    """
    Genera un diccionario de stock acumulado a partir de los pasillos disponibles.

    Parámetros:
    - pasillos_list: lista de diccionarios {item_id: cantidad}, donde cada diccionario representa un pasillo.

    Retorna:
    - stock: diccionario {item_id: cantidad total disponible}.
    """

    stock = {}  # Diccionario para acumular el stock total

    # Iterar sobre todos los pasillos y sumar cantidades por item_id
    for pasillo in pasillos_list:
        for item_id, cantidad in pasillo.items():
            stock[item_id] = stock.get(item_id, 0) + cantidad

    return stock

def pasillos(pasillos_list, demanda, num_pasillos):
    """
    Minimiza la cantidad de pasillos requeridos para satisfacer la demanda de ítems.

    Parámetros:
    - pasillos_list: lista de diccionarios {item_id: cantidad}, donde cada diccionario representa un pasillo.
    - demanda: diccionario {item_id: cantidad requerida}.

    Retorna:
    - valor_objetivo: número mínimo de pasillos requeridos.
    - pasillos_seleccionados: lista de índices de pasillos utilizados.
    """

    # Definir el problema de optimización
    prob = pulp.LpProblem("Minimize_Pasillos", pulp.LpMinimize)

    # Variables de decisión: si cada pasillo se usa (binario)
    y = {j: pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(num_pasillos)}

    # Función objetivo: minimizar número de pasillos seleccionados
    prob += pulp.lpSum(y[j] for j in range(num_pasillos))

    # Restricciones de demanda
    for item_id, cantidad_requerida in demanda.items():
        prob += pulp.lpSum(pasillos_list[j].get(item_id, 0) * y[j] for j in range(num_pasillos)) >= cantidad_requerida

    # Resolver el problema
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Obtener resultados
    pasillos_seleccionados = [j for j in range(num_pasillos) if y[j].varValue > 0.5]
    n_pasillos = len(pasillos_seleccionados)

    return n_pasillos, pasillos_seleccionados

def funcion_objetivo(demanda, n_pasillos, limite_inferior, limite_superior, exceso_stock, penalizacion=10**3):
    """
    Calcula la función objetivo para minimizar pasillos con penalizaciones ajustadas.

    Parámetros:
    - demanda: diccionario {item_id: cantidad total requerida}.
    - n_pasillos: número mínimo de pasillos seleccionados.
    - limite_inferior: umbral inferior de demanda permitida.
    - limite_superior: umbral superior de demanda permitida.
    - exceso_stock: exceso de demanda sobre el stock disponible.
    - penalizacion: factor de penalización por exceso de restricciones.

    Retorna:
    - Valor de la función objetivo ajustada.
    """

    sum_i = sum(demanda.values())
    
    # Penalización por límites
    if sum_i < limite_inferior:
        penalizacion_valor = (limite_inferior - sum_i) / limite_inferior
    elif sum_i > limite_superior:
        penalizacion_valor = (sum_i - limite_superior) / limite_superior
    else:
        penalizacion_valor = 0
    # Penalización por exceso de stock
    penalizacion_stock = min(exceso_stock / sum_i, 1) if sum_i > 0 else 1

    # Calcular función objetivo
    if n_pasillos == 0:  # Evitar división por cero
        return -penalizacion * (penalizacion_valor + penalizacion_stock)
    
    fun = (sum_i / n_pasillos) - penalizacion * (penalizacion_valor + penalizacion_stock)
    
    return fun