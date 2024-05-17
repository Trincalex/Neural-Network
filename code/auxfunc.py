'''

    auxfunc.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene alcune funzionalità aggiuntive per la creazione
    della rete neurale e l'esecuzione del programma, tra cui la definizione
    delle funzioni di attivazione e delle funzioni di errore.

'''

# ########################################################################### #
# LIBRERIE

import numpy as np
import math

# ########################################################################### #
# FUNZIONI DI ATTIVAZIONE

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

def sum_of_squares(
        prediction : np.ndarray, 
        target : np.ndarray,
        der : bool = False
) -> float | np.ndarray:
    
    """

        È una funzione di errore tipicamente utilizzata per i problemi di regressione.

        Parameters:
        -   prediction: è l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: è l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto al target.

        Returns:
        -   se der=False, restituisce la somma dei quadrati degli errori componente per componente.
        -   se der=True, invece, restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto al target.
    
    """

    # Calcolo delle distanze tra predizioni e target (errori)
    errors = prediction - target
    # print(errors)

    """
    Sia 'n' la lunghezza dei vettori in output.
    Il valore da restituire in output dovrebbe essere diviso per 'n' per poterne calcolare la media ed ottenere dei risultati più consistenti che non dipendono da questa dimensione.
    In realtà, però, si può dividere per una qualsiasi costante, siccome questo non modifica la convessita' della funzione.
    In particolare, scegliamo 2: in questo modo, nell'andare a calcolare la derivata prima, tale prodotto si cancella e rende i calcoli successivi più semplici e leggibili.
    """

    if der:
        # Nel calcolare la derivata prima, il prodotto (1/2) * 2 si cancella
        # Per calcolare la matrice jacobiana, non restituiamo la somma ma l'intero vettore.
        return errors
    
    return np.sum(errors ** 2) / 2

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
) -> float | np.ndarray:
    
    """

        È una funzione di errore tipicamente utilizzata per i problemi di classificazione.

        Parameters:
        -   prediction: è l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: è l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto al target.

        Returns:
        -   se der=False, restituisce la somma dei quadrati degli errori componente per componente.
        -   se der=True, invece, ne restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto al target.
    
    """
    
    z = soft_max(prediction)

    if der:
        return z - target
    
    return -np.sum(target * np.log(z))

def split_dataset(dataset,labels,k):
    k_fold_dataset = np.array_split(dataset,k)
    k_fold_labels = np.array_split(labels,k)
    return k_fold_dataset,k_fold_labels

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
# https://www.quora.com/Why-is-the-sum-of-squares-in-linear-regression-divided-by-2n-where-n-is-the-number-of-observations-instead-of-just-n
# https://www.youtube.com/watch?v=f50tlks5caI