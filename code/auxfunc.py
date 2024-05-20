'''

    auxfunc.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene alcune funzionalita' aggiuntive per la creazione
    della rete neurale e l'esecuzione del programma, tra cui la definizione
    delle funzioni di attivazione e delle funzioni di errore.

'''

# ########################################################################### #
# LIBRERIE

import numpy as np
import math
import constants

# ########################################################################### #
# FUNZIONI DI ATTIVAZIONE

def leaky_relu(input : float, der : bool = False) -> float:
    """

        Calcola il valore di attivazione di un neurone utilizzando un miglioramento della classica ReLU (Rectified Linear Unit), in quanto la sua derivata prima e' sempre un valore non negativo.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'attivazione della Leaky ReLU.
        -   se der=True, invece, ne restituisce il gradiente.
    
    """

    if der:
        return np.where(input >= 0, 1, constants.LEAKY_RELU_ALPHA)

    return np.where(input >= 0, input, constants.LEAKY_RELU_ALPHA * input)

# end

def sigmoid(input : float, der : bool = False) -> float:
    """

        Calcola il valore di attivazione di un neurone utilizzando la funzione logistica.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'attivazione della sigmoide.
        -   se der=True, invece, ne restituisce il gradiente.
    
    """
    
    fx = 1 + math.exp(-input)

    if der:
        return (-math.exp(-input)) / (-math.pow(fx, 2))
    
    return 1 / fx

# end

def tanh(input : float, der : bool = False) -> float:
    """

        Calcola il valore di attivazione di un neurone utilizzando la tangente iperbolica.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'attivazione della tangente iperbolica.
        -   se der=True, invece, ne restituisce il gradiente.
    
    """
    
    sinh = (math.exp(input) - math.exp(-input)) / 2
    cosh = (math.exp(input) + math.exp(-input)) / 2

    if der:
        return 1 / (cosh ** 2)
    
    return sinh / cosh

# end
    
def identity(input : float, der : bool = False) -> float:
    """

        Calcola il valore di attivazione di un neurone utilizzando la funzione identita'.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'input.
        -   se der=True, invece, ne restituisce il gradiente, cioe' 1.
    
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

        E' una funzione di errore tipicamente utilizzata per i problemi di regressione.

        Parameters:
        -   prediction: e l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: e l'etichetta di classificazione di una determinata coppia del dataset.
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
    Il valore da restituire in output dovrebbe essere diviso per 'n' per poterne calcolare la media ed ottenere dei risultati piu' consistenti che non dipendono da questa dimensione.
    In realta', pero', si puo' dividere per una qualsiasi costante, siccome questo non modifica la convessita' della funzione.
    In particolare, scegliamo 2: in questo modo, nell'andare a calcolare la derivata prima, tale prodotto si cancella e rende i calcoli successivi piu' semplici e leggibili.
    """

    if der:
        # Nel calcolare la derivata prima, il prodotto (1/2) * 2 si cancella
        # Per calcolare la matrice jacobiana, non restituiamo la somma ma l'intero vettore.
        return errors
    
    return np.sum(errors ** 2) / 2

# end

def softmax(prediction : np.ndarray) -> np.ndarray:

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

        E' una funzione di errore tipicamente utilizzata per i problemi di classificazione.

        Parameters:
        -   prediction: e l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: e l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto al target.

        Returns:
        -   se der=False, restituisce la somma dei quadrati degli errori componente per componente.
        -   se der=True, invece, ne restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto al target.
    
    """
    
    z = softmax(prediction)

    if der:
        return z - target
    
    return -np.sum(target * np.log(z))

# end

def split_dataset(dataset : np.array ,labels : np.array ,k : int):
    """
        Divide il dataset e le label in k sottoinsiemi

        Parameters:
        -   dataset: il dataset da dividere
        -   labels: i valori target del dataset
        -   k: il numero di insiemi in cui dividere i dataset e le labels
        
        Returns:
        -   k_fold_dataset: array che contiene i k array in cui e' stato suddiviso il dataset
        -   k_fold: array che contiene i k array in cui sono state divise le labels
    """

    k_fold_dataset = np.array_split(dataset,k)
    k_fold_labels = np.array_split(labels,k)
    return k_fold_dataset, k_fold_labels

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
# https://www.digitalocean.com/community/tutorials/relu-function-in-python