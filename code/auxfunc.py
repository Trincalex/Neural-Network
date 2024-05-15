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
import math

# ########################################################################### #
# FUNZIONI DI ATTIVAZIONE

# def sigmoid(input : float, der : bool = False) -> float:
#     out = 1 / (1 + np.exp(-input))
#     if der:
#         return out * (1 - out)
#     return out
# # end

def sigmoid(input : float, der : bool = False) -> float:

    """

        ...

        Parameters:
        -   

        Returns:
        -   
    
    """
    
    fx = 1 + math.exp(-input)

    if der:
        return (-math.exp(-input)) / (-math.pow(fx, 2))
    return 1 / fx

# end

# def tanh(input : float, der : bool = False) -> float:
#     temp = np.exp(2 * input)
#     out = (temp - 1) / (temp + 1)
#     if der:
#         return 1 - out * out
#     return out
# # end

def tanh(input : float, der : bool = False) -> float:

    """

        ...

        Parameters:
        -   

        Returns:
        -   
    
    """
    
    sinh = (math.exp(input) - math.exp(-input)) / 2
    cosh = (math.exp(input) + math.exp(-input)) / 2

    if der:
        return 1 / (cosh ** 2)
    return sinh / cosh

# end
    
def identity(input : float, der : bool = False) -> float:

    """

        ...

        Parameters:
        -   

        Returns:
        -   
    
    """
    
    if der:
        return 1
    return input

# end

# ########################################################################### #
# FUNZIONI DI ERRORE
    
# def sum_of_square(x : np.ndarray, t : np.ndarray, der : bool = False) -> float:
#     z = x + t
#     if der:
#         return z
#     return (1/2) * np.sum(np.power(z, 2))
# # end

def sum_of_square(
        prediction : np.ndarray, 
        target : np.ndarray,
        der : bool = False
) -> float:
    
    """

        ...

        Parameters:
        -   

        Returns:
        -   
    
    """
    
    # print(target)
    # print(prediction)

    errors = prediction - target
    print(errors)

    length = len(errors)
    # print(length)

    # sum = np.sum(errors ** 2)
    # print(sum)

    # mean = sum / length
    # print(mean)

    if der:
        return 2 * np.sum(errors) / length
    
    return np.sum(errors ** 2) / length

# end

def soft_max(prediction : np.ndarray) -> np.ndarray:

    """

        ...

        Parameters:
        -   

        Returns:
        -   
    
    """
    
    x_exp = np.exp(prediction - prediction.max(0))
    z = x_exp / np.sum(x_exp, 0)
    return z

# end

def cross_entropy(
        prediction : np.ndarray,
        target : np.ndarray,
        der : bool = False
) -> float:
    
    """

        ...

        Parameters:
        -   

        Returns:
        -   
    
    """
    
    z = soft_max(prediction)

    if der:
        return z - target
    
    # return -(target*np.log(z)).sum()
    return -np.sum(target * np.log(z))

# end

# ########################################################################### #
# ALTRE FUNZIONI

# def stampa_matrice(matrice):
#     for riga in matrice:
#         for elemento in riga:
#             print(int(elemento), end=" ")
#         print()

# ########################################################################### #
# RIFERIMENTI

# https://www.matematika.it/public/allegati/33/Grafici_domini_derivate_funzioni_iperboliche_1_1.pdf
# https://www.matematika.it/public/allegati/33/11_44_Derivate_2_3.pdf