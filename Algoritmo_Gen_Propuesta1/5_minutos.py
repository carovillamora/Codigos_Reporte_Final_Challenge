#!/usr/bin/python3

import funciones_entero as fn
import genetico_entero as gn
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--mu", type=int, required=True)
    parser.add_argument("--select", type=str, required=True)
    parser.add_argument("--pc", type=float, required=True)
    parser.add_argument("--recom", type=str, required=True)
    parser.add_argument("--pm", type=float, required=True)
    args = parser.parse_args()

    start_time = time.time()

    while time.time() - start_time < 300:

        general, ordenes_list, pasillos_list = fn.lectura(args.instance)

        stock = fn.generar_stock(pasillos_list)
        S = gn.inicio(args.mu, general, ordenes_list, stock, pasillos_list)
        
        i = 0
        while time.time() - start_time < 300:

            M = gn.seleccion(S, args.mu, args.select)
            p_prima = gn.recombinacion(M, args.mu, args.pc, args.recom)
            p_prima = gn.mutacion(p_prima, args.mu, args.pm, ordenes_list, pasillos_list, general, stock)
            S = gn.reemplazo(S, p_prima, args.mu)
            i += 1

    print(S[0])

if __name__ == "__main__":
    main()
