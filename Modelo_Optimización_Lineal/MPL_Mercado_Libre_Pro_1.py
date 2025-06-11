import numpy as np
from docplex.mp.model import Model
import copy
import time

# FUNCIÓN CREADA POR CARO Y BRYAN
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

    num_ordenes, num_items, num_pasillos = general[0], general[1], general[2]

    matriz_ordenes = np.zeros((num_ordenes,num_items), dtype=int)

    for i, orden in enumerate(datos[:num_ordenes]):
        for j in range(1, len(orden), 2):
            cantidad = orden[j+1]
            matriz_ordenes[i][orden[j]]=cantidad

    matriz_pasillos = np.zeros((num_pasillos,num_items), dtype=int)

    for i, pasillo in enumerate(datos[num_ordenes:]):
        for j in range(1, len(pasillo), 2):
            cantidad = pasillo[j+1]
            matriz_pasillos[i][pasillo[j]] = cantidad

    return general, matriz_ordenes, matriz_pasillos
#return cantidad_ordenes, numero_pasillos, numero_items, liminf, limsup, matriz_ordenes, matriz_pasillos

################################################################################
# PROPUESTA DE AXEL
# PRIMERA PROPUESTA DE MODELO
# Usando el planteamiento del problema sin límite en la cantidad de pasillos

# Obtención de los datos de la instancia, O,A,I,LB,UB,UO,UA
"""
O,A,I=5,5,5         # Instancia 1
LB, UB = 1, 18
UO = [[3,0,1,0,0],
      [0,1,0,1,0],
      [0,0,1,0,2],
      [1,0,2,1,1],
      [0,1,0,0,0]]  # Elementos de los pedidos
UA = [[2,1,1,0,1],
      [2,1,2,0,1],
      [0,2,0,1,2],
      [2,1,0,1,1],
      [0,1,2,1,2]]  # Elementos en los pasillos
"""

#ruta_prueba="Instancias/instance_0001.txt"
#ruta_prueba="Instancias/instance_0002.txt"
#ruta_prueba="Instancias/instance_0003.txt"
#ruta_prueba="Instancias/instance_0004.txt"
#ruta_prueba="Instancias/instance_0005.txt"
#ruta_prueba="Instancias/instance_0006.txt"
#ruta_prueba="Instancias/instance_0007.txt"
#ruta_prueba="Instancias/instance_0008.txt"
#ruta_prueba="Instancias/instance_0009.txt"
#ruta_prueba="Instancias/instance_0010.txt"
#ruta_prueba="Instancias/instance_0011.txt"
#ruta_prueba="Instancias/instance_0012.txt"
#ruta_prueba="Instancias/instance_0013.txt"
ruta_prueba="Instancias/instance_0014.txt"
#ruta_prueba="Instancias/instance_0015.txt"
#ruta_prueba="Instancias/instance_0016.txt"
#ruta_prueba="Instancias/instance_0017.txt"
#ruta_prueba="Instancias/instance_0018.txt"
#ruta_prueba="Instancias/instance_0019.txt"
#ruta_prueba="Instancias/instance_0020.txt"

start = time.time()
general,UO,UA = lectura(ruta_prueba)
O,I,A,LB,UB = general[0], general[1], general[2], general[3], general[4]
end = time.time()
tiempo_lectura = end-start


# Creación de la matriz de valores p_oi del modelo de optimización no lineal
PO=copy.deepcopy(UO)
for o in range(O):
    for i in range(I):
        if UO[o][i]>0: PO[o][i] = 1

# Creación del vectorde valores B_o del modelo de optimización no lineal
B=[sum(PO[o][i]*UO[o][i] for i in range(I)) for o in range(O)]
# M_big = Límite superior de la variable t para la linealización del producto de variables
M_big = sum(abs(UO[o][i]) for i in range(I) for o in range(O))
#print(PO)
#print(B)
#print(M_big)

# Crear el modelo de programación con CPLEX
# Modelo de administración de waves de Mercado Libre
m = Model("Modelo de administración de waves de Mercado Libre")

# Dado un piso en un momento dado con I elementos repartidos en A pasillos que buscan satisfacer las órdenes O
#  y_a = 1 si se va a pasar por el pasillo a; 0 en caso contrario
#  z_0 = 1 si se va a completar la orden o; 0 en caso contrario
# a=1,...,A; o=1,...,O
s = m.continuous_var_list(keys=A,lb=0,name="s")
w = m.continuous_var_list(keys=O,lb=0,name="w")
y = m.binary_var_list(keys=A,lb=0,name="y")
z = m.binary_var_list(keys=O,lb=0,name="z")
t = m.continuous_var(lb=0, ub=M_big, name="t")

# Restricciones
# Límites inferior y superior de la cantidad de elementos a tomar en la wave
m.add_constraint(m.sum(B[o]*w[o] for o in range(O)) >= LB*t)
m.add_constraint(m.sum(B[o]*w[o] for o in range(O)) <= UB*t)

# No elegir ordenes que sobrepasen la cantidad de inventario en el piso por elemento
m.add_constraints(m.sum(UO[o][i]*w[o] for o in range(O)) <= m.sum(UA[a][i]*s[a] for a in range(A)) for i in range(I))

# Definición de la variable t
m.add_constraint(m.sum(s[a] for a in range(A)) == 1)

# Linealización del producto w = z*t
m.add_constraints(w[o] <= M_big*z[o] for o in range(O))
m.add_constraints(w[o] <= t for o in range(O))
m.add_constraints(w[o] >= t-M_big*(1-z[o]) for o in range(O))

# Linealización del producto s = y*t
m.add_constraints(s[a] <= M_big*y[a] for a in range(A))
m.add_constraints(s[a] <= t for a in range(A))
m.add_constraints(s[a] >= t-M_big*(1-y[a]) for a in range(A))

# Función objetivo
obj=m.sum(m.sum(B[o]*w[o] for o in range(O)))
m.maximize(obj)


# CONFIGURACIONES DEL SOLVER
# Configurar para que CPLEX busque soluciones factibles rápidamente
m.context.cplex_parameters.emphasis.mip = 1  # 1 = Prioriza factibilidad
# Límite de memoria de trabajo (en MB)
m.context.cplex_parameters.workmem = 8192  # 8 GB en MB
# Minimizar el uso del disco
#m.context.cplex_parameters.memoryemphasis = 1
# Configurar el tiempo máximo de ejecución
m.set_time_limit(600)  # 10 minutos 


start = time.time()
solution = m.solve(log_output=True,)
end = time.time()
tiempo_busqueda = end-start
print(f"\nTiempo total para la lectura del archivo: {tiempo_lectura} segundos\n")
print(f"\nTiempo total para la resolución del problema: {tiempo_busqueda} segundos")

# Impresión de resultados
# Impresión fitness
numerador = sum(B[o]*z[o].solution_value for o in range(O))
denominador = sum(y[a].solution_value for a in range(A))
objetivo = numerador/denominador

print(f"\n\tNúmero de elementos recolectados por pasillo visitado: {objetivo}")
#print(f"Número de elementos recolectados: {numerador}")
#print(f"Número de pasillos visitados: {denominador}")

#print("Pedidos a completarse en el Wave:")
#for o in range(O):
#    if z[o].solution_value>0:
#        print(f"Pedido no. {o}")

#print("Pasillos por los que se van a pasar en el Wave:")
#for a in range(A):
#    if y[a].solution_value>0 and a<5:
#        print(f"Pasillo no. {a}")

#print(UO[0])

""""""
