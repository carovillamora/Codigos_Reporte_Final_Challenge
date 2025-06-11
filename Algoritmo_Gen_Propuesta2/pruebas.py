import random
import numpy as np

# Lectura del archivo
def lectura(archivo: str):
    with open(archivo) as info:
        datos = [list(map(int, linea.split())) for linea in info if linea.strip()]
    print(datos)

    # Extracción de datos
    cantidad_ordenes = datos[0][0]  
    numero_items = datos[0][1]
    numero_pasillos = datos[0][2] 
    #print(cantidad_ordenes, numero_items, numero_pasillos)
    del datos[0]

    #print(datos)

    limites = datos.pop()
    liminf = limites[0]
    limsup = limites[1]
    #print(liminf, limsup)

    #print(datos)
    #print(len(datos))  

    matriz_ordenes = np.zeros((cantidad_ordenes,numero_items), dtype=int)

    for i in range(cantidad_ordenes):
        orden = datos[i]
        del datos[i]
        for o in range(1,len(orden),2):
            matriz_ordenes[i][orden[o]]=orden[o+1]

    np.set_printoptions(threshold=np.inf)
    #print(matriz_ordenes)
    #print(datos)

    matriz_pasillos = np.zeros((numero_pasillos,numero_items), dtype=int)

    for j in range(numero_pasillos):
        pasillo = datos[j]
        for a in range(1,len(pasillo),2):
            matriz_pasillos[j][pasillo[a]]=pasillo[a+1]

    np.set_printoptions(threshold=np.inf)
    #print(matriz_pasillos)

    return  matriz_pasillos, matriz_ordenes, liminf, limsup

