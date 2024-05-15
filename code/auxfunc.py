'''

    auxfunc.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene alcune funzionalità aggiuntive per la creazione
    della rete neurale e l'esecuzione del programma.

'''

# ########################################################################### #
# LIBRERIE

import numpy as np

# ########################################################################### #
# FUNZIONI DI ATTIVAZIONE

def sigmoid(x, der=False):
    y = 1 / (1 + np.exp(-x))
    if der:
        return y * (1 - y)
    return y
# end

def tanh(x, der=False):
    a = np.exp(2 * x)
    y = (a - 1) / (a + 1)
    if der:
        return 1 - y * y
    return y
# end
    
def identity(x, der=False):
    if der:
        return 1
    return x
# end

# ########################################################################### #
# FUNZIONI DI ERRORE
    
def sum_of_square(x, t, der=False):
    z = x + t
    if der:
        return z
    return (1/2) * np.sum(np.power(z, 2))
# end

def soft_max(x):
    x_exp = np.exp(x - x.max(0))
    z = x_exp / np.sum(x_exp, 0)
    return z
# end

def cross_entropy(x, t, der=False):
    z = soft_max(x)
    if der:
        return z-t
    return -(t*np.log(z)).sum()
# end

# ########################################################################### #
# ALTRE FUNZIONI

def stampa_matrice(matrice):
    for riga in matrice:
        for elemento in riga:
            print(int(elemento), end=" ")
        print()

# ########################################################################### #
# RIFERIMENTI

