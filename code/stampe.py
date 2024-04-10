import numpy as np

def stampa_matrice(matrice):
    for riga in matrice:
        for elemento in riga:
            print(int(elemento), end=" ")
        print()