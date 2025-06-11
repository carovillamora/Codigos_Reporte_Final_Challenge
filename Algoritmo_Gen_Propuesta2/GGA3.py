import sys
import random
import numpy as np
from Fitness import fitness
from pruebas import matriz_ordenes, matriz_pasillos, liminf, limsup

class AlgoritmoGenetico:
    def __init__(self, archivo=None, generaciones=100, tam_poblacion=50, prob_cruce=0.8, prob_mut=0.1):
        self.archivo = archivo
        self.generaciones = generaciones
        self.tam_poblacion = tam_poblacion
        self.prob_cruce = prob_cruce
        self.prob_mut = prob_mut
        self.mejor_solucion = None
        self.mejor_fitness = -np.inf

    def obtener_pasillos_para_items(self, items):
        """Encuentra los pasillos que contienen los items especificados"""
        pasillos_necesarios = set()
        for item in items:
            for pasillo_idx, pasillo in enumerate(matriz_pasillos):
                if item in pasillo:
                    pasillos_necesarios.add(pasillo_idx)
                    break
        return list(pasillos_necesarios)

    def generar_solucion_aleatoria(self):
        """Genera una solución completamente aleatoria"""
        num_ordenes = random.randint(1, len(matriz_ordenes))
        ordenes = random.sample(range(len(matriz_ordenes)), num_ordenes)
        
        items = []
        for orden in ordenes:
            items.extend([i for i, cant in enumerate(matriz_ordenes[orden]) if cant > 0])
        
        pasillos = self.obtener_pasillos_para_items(items)
        return [num_ordenes] + ordenes + [len(pasillos)] + pasillos

    def generar_solucion_heuristica(self):
        """Genera una solución usando un enfoque heurístico simple"""
        ordenes_con_items = [(i, sum(1 for cant in matriz_ordenes[i] if cant > 0)) 
                            for i in range(len(matriz_ordenes))]
        ordenes_ordenadas = sorted(ordenes_con_items, key=lambda x: -x[1])
        
        ordenes_seleccionadas = []
        unidades = 0
        for orden, _ in ordenes_ordenadas:
            if unidades >= limsup:
                break
            ordenes_seleccionadas.append(orden)
            unidades += sum(matriz_ordenes[orden])
        
        while unidades < liminf and len(ordenes_seleccionadas) < len(matriz_ordenes):
            for orden, _ in ordenes_ordenadas:
                if orden not in ordenes_seleccionadas:
                    ordenes_seleccionadas.append(orden)
                    unidades += sum(matriz_ordenes[orden])
                    break
        
        while unidades > limsup and len(ordenes_seleccionadas) > 1:
            orden_a_quitar = min(ordenes_seleccionadas, 
                               key=lambda o: sum(1 for cant in matriz_ordenes[o] if cant > 0))
            ordenes_seleccionadas.remove(orden_a_quitar)
            unidades -= sum(matriz_ordenes[orden_a_quitar])
        
        items = []
        for orden in ordenes_seleccionadas:
            items.extend([i for i, cant in enumerate(matriz_ordenes[orden]) if cant > 0])
        pasillos = self.obtener_pasillos_para_items(items)
        
        return [len(ordenes_seleccionadas)] + ordenes_seleccionadas + [len(pasillos)] + pasillos

    def generar_solucion_valida(self):
        """Genera una solución válida que cumple con todas las restricciones"""
        if random.random() < 0.5:
            sol = self.generar_solucion_heuristica()
            if liminf <= sum(sum(matriz_ordenes[o]) for o in sol[1:1+sol[0]]) <= limsup:
                return sol
        
        for _ in range(100):
            sol = self.generar_solucion_aleatoria()
            unidades = sum(sum(matriz_ordenes[o]) for o in sol[1:1+sol[0]])
            if liminf <= unidades <= limsup and sol[1+sol[0]] > 0:
                return sol
        
        for orden in range(len(matriz_ordenes)):
            unidades = sum(matriz_ordenes[orden])
            if liminf <= unidades <= limsup:
                items = [i for i, cant in enumerate(matriz_ordenes[orden]) if cant > 0]
                pasillos = self.obtener_pasillos_para_items(items)
                if pasillos:
                    return [1, orden, len(pasillos)] + pasillos
        
        return [1, 0, 1, 0]

    def reparar_solucion(self, sol):
        """Repara una solución para que cumpla con todas las restricciones"""
        try:
            num_ordenes = sol[0]
            ordenes = sol[1:1+num_ordenes]
            num_pasillos = sol[1+num_ordenes] if len(sol) > 1+num_ordenes else 0
            pasillos = sol[2+num_ordenes:2+num_ordenes+num_pasillos] if len(sol) > 2+num_ordenes else []
            
            ordenes = list({o for o in ordenes if 0 <= o < len(matriz_ordenes)})
            num_ordenes = len(ordenes)
            
            items = []
            for orden in ordenes:
                items.extend([i for i, cant in enumerate(matriz_ordenes[orden]) if cant > 0])
            
            pasillos = self.obtener_pasillos_para_items(items)
            num_pasillos = len(pasillos)
            
            unidades = sum(sum(matriz_ordenes[o]) for o in ordenes)
            intentos = 0
            
            while (unidades < liminf or unidades > limsup) and intentos < 100:
                if unidades < liminf and num_ordenes < len(matriz_ordenes):
                    disponibles = [i for i in range(len(matriz_ordenes)) if i not in ordenes]
                    if disponibles:
                        ordenes.append(random.choice(disponibles))
                        num_ordenes += 1
                elif unidades > limsup and num_ordenes > 1:
                    ordenes.pop(random.randint(0, num_ordenes-1))
                    num_ordenes -= 1
                
                items = []
                for orden in ordenes:
                    items.extend([i for i, cant in enumerate(matriz_ordenes[orden]) if cant > 0])
                pasillos = self.obtener_pasillos_para_items(items)
                num_pasillos = len(pasillos)
                unidades = sum(sum(matriz_ordenes[o]) for o in ordenes)
                intentos += 1
            
            if num_pasillos > 0 and liminf <= unidades <= limsup:
                return [num_ordenes] + ordenes + [num_pasillos] + pasillos
        except:
            pass
        
        return self.generar_solucion_valida()

    def seleccion_ranking(self, poblacion, fitnesses, num_padres):
        """Selección por ranking lineal"""
        ranked = sorted(zip(poblacion, fitnesses), key=lambda x: -x[1])
        probabilidades = [i/len(ranked) for i in range(1, len(ranked)+1)]
        return random.choices(
            [ind for ind, fit in ranked],
            weights=probabilidades,
            k=num_padres
        )

    def seleccion_diversidad(self, poblacion, fitnesses, num_padres):
        """Selección basada en diversidad de pasillos"""
        padres = []
        mejor_ind = poblacion[np.argmax(fitnesses)]
        padres.append(mejor_ind.copy())
        
        for _ in range(num_padres - 1):
            if len(padres) == 0:
                elegido = random.choice(poblacion)
            else:
                pasillos_cubiertos = set()
                for ind in padres:
                    num_ordenes = ind[0]
                    pasillos_ind = ind[2+num_ordenes:2+num_ordenes+ind[1+num_ordenes]]
                    pasillos_cubiertos.update(pasillos_ind)
                
                diversidades = []
                for ind in poblacion:
                    num_ordenes = ind[0]
                    pasillos_ind = ind[2+num_ordenes:2+num_ordenes+ind[1+num_ordenes]]
                    nuevos_pasillos = len(set(pasillos_ind) - pasillos_cubiertos)
                    diversidades.append(nuevos_pasillos + 1)
                
                elegido = random.choices(poblacion, weights=diversidades, k=1)[0]
            
            padres.append(elegido.copy())
        
        return padres

    def ejecutar(self):
        """Ejecuta el algoritmo genético completo"""
        poblacion = []
        for i in range(self.tam_poblacion):
            if i < self.tam_poblacion // 2:
                poblacion.append(self.generar_solucion_heuristica())
            else:
                poblacion.append(self.generar_solucion_valida())
        
        for generacion in range(self.generaciones):
            fitnesses = [fitness(ind, matriz_ordenes, matriz_pasillos) for ind in poblacion]
            
            max_fit = max(fitnesses)
            if max_fit > self.mejor_fitness:
                mejor_idx = np.argmax(fitnesses)
                self.mejor_solucion = poblacion[mejor_idx].copy()
                self.mejor_fitness = max_fit
            
            # Selección
            num_padres_ranking = int(self.tam_poblacion * 0.25)
            padres_ranking = self.seleccion_ranking(poblacion, fitnesses, num_padres_ranking)
            padres_diversidad = self.seleccion_diversidad(poblacion, fitnesses, 
                                                        self.tam_poblacion - num_padres_ranking)
            padres = padres_ranking + padres_diversidad
            
            # Cruce
            descendencia = []
            for i in range(0, len(padres), 2):
                if i+1 < len(padres):
                    padre1, padre2 = padres[i], padres[i+1]
                    min_len = min(len(padre1), len(padre2))
                    
                    if min_len > 2:
                        punto1 = random.randint(1, min_len-2)
                        punto2 = random.randint(punto1+1, min_len-1)
                        hijo1 = padre1[:punto1] + padre2[punto1:punto2] + padre1[punto2:]
                        hijo2 = padre2[:punto1] + padre1[punto1:punto2] + padre2[punto2:]
                    else:
                        punto = random.randint(1, min_len-1)
                        hijo1 = padre1[:punto] + padre2[punto:]
                        hijo2 = padre2[:punto] + padre1[punto:]
                    
                    descendencia.extend([self.reparar_solucion(hijo1), 
                                       self.reparar_solucion(hijo2)])
                else:
                    descendencia.append(padres[i].copy())
            
            # Mutación
            for i in range(len(descendencia)):
                if random.random() < self.prob_mut:
                    mutado = descendencia[i].copy()
                    if len(mutado) > 4:
                        idx = random.randint(1, len(mutado)-1)
                        mutado[idx] = random.choice(range(len(matriz_pasillos)))
                    descendencia[i] = self.reparar_solucion(mutado)
            
            # Reemplazo
            poblacion = descendencia
        
        # Preparar resultados
        num_ordenes = self.mejor_solucion[0]
        ordenes = self.mejor_solucion[1:1+num_ordenes]
        pasillos = self.mejor_solucion[2+num_ordenes:2+num_ordenes+self.mejor_solucion[1+num_ordenes]]
        
        return {
            'ordenes': ordenes,
            'num_ordenes': num_ordenes,
            'pasillos': pasillos,
            'num_pasillos': len(pasillos),
            'unidades': sum(sum(matriz_ordenes[o]) for o in ordenes),
            'items_unicos': len(set(i for o in ordenes for i, cant in enumerate(matriz_ordenes[o]) if cant > 0)),
            'fitness': self.mejor_fitness,
            'parametros': {
                'generaciones': self.generaciones,
                'tam_poblacion': self.tam_poblacion,
                'prob_cruce': self.prob_cruce,
                'prob_mut': self.prob_mut
            }
        }

def algoritmo_genetico_mejorado(archivo=None, generaciones=100, tam_poblacion=50, prob_cruce=0.8, prob_mut=0.1):
    """Función wrapper para compatibilidad con target-runner.py"""
    # Si se ejecuta desde target-runner.py, los parámetros vendrán en sys.argv
    #if len(sys.argv) > 1:
        # El primer argumento es el nombre del script, los siguientes son los parámetros
        # Ejemplo de llamada desde target-runner.py: ./GGA.py INSTANCE generaciones tam_poblacion prob_cruce prob_mut SEED
        #if len(sys.argv) >= 7:
        #    archivo = sys.argv[1]  # INSTANCE (nombre del archivo de instancia)
        #    generaciones = int(sys.argv[2])
        #    tam_poblacion = int(sys.argv[3])
        #    prob_cruce = float(sys.argv[4])
        #    prob_mut = float(sys.argv[5])
        #    seed = int(sys.argv[6])  # Semilla para reproducibilidad
        #    random.seed(seed)
        #    np.random.seed(seed)
    
    ag = AlgoritmoGenetico(
        archivo=archivo,
        generaciones=generaciones,
        tam_poblacion=tam_poblacion,
        prob_cruce=prob_cruce,
        prob_mut=prob_mut
    )
    return ag.ejecutar()

if __name__ == "__main__":
    # Si se ejecuta desde target-runner.py, solo imprime el fitness
    if len(sys.argv) > 1:
        resultado = algoritmo_genetico_mejorado()
        print(resultado['fitness'])
        print("\nMejor solución encontrada:")
        print(f"Órdenes seleccionadas: {resultado['selected_orders']}")
        print(f"Número de órdenes: {resultado['num_orders']}")
        print(f"Unidades totales: {resultado['total_units']} (LB={resultado['LB']}, UB={resultado['UB']})")
        print(f"Ítems únicos recogidos: {resultado['unique_items']}")
        print(f"Pasillos visitados: {resultado['visited_aisles']}")
        print(f"Fitness (Ítems/Pasillos): {resultado['fitness']:.2f}")
        
    else:
        # Ejemplo de ejecución directa (modo normal)
        resultado = algoritmo_genetico_mejorado(
            generaciones=100,
            tam_poblacion=50,
            prob_cruce=0.8,
            prob_mut=0.1
        )
        
        print("\nMejor solución encontrada:")
        print(f"Órdenes seleccionadas: {resultado['selected_orders']}")
        print(f"Número de órdenes: {resultado['num_orders']}")
        print(f"Unidades totales: {resultado['total_units']} (LB={resultado['LB']}, UB={resultado['UB']})")
        print(f"Ítems únicos recogidos: {resultado['unique_items']}")
        print(f"Pasillos visitados: {resultado['visited_aisles']}")
        print(f"Fitness (Ítems/Pasillos): {resultado['fitness']:.2f}")
