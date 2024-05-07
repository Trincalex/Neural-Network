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

def sigmoid(x,der=0):
    y=1/(1+np.exp(-x))
    if der==0:
        return y
    else:
        return y*(1-y)

def tanh(x,der=0):
    a=np.exp(2*x)
    y=(a-1)/(a+1)
    if der==0:
        return y
    else:
        return 1-y*y
    
def identity(x,der=0):
    if der==0:
        return x
    else:
        return 1

# ########################################################################### #
# FUNZIONI DI ERRORE
    
def sum_of_square (x,t,der=0):
    z = x+t
    if der==0:
        return (1/2)*np.sum(np.power(z,2))
    else:
        return z

def soft_max(x):
    x_exp=np.exp(x-x.max(0))
    z=x_exp/np.sum(x_exp,0)
    return z

def cross_entropy(x,t,der=0):
    z=soft_max(x)
    if der==0:
        return -(t*np.log(z)).sum()
    else:
        return z-t

# ########################################################################### #
# ALTRE FUNZIONI

def stampa_matrice(matrice):
    for riga in matrice:
        for elemento in riga:
            print(int(elemento), end=" ")
        print()

# ########################################################################### #
# RIFERIMENTI

