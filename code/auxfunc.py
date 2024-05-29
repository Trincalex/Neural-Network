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
        return np.where(input >= 0, 1, constants.DEFAULT_LEAKY_RELU_ALPHA)

    return np.where(input >= 0, input, constants.DEFAULT_LEAKY_RELU_ALPHA * input)

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

    try:
        fx = 1 + np.exp(-input)
    except (OverflowError, RuntimeWarning):
        # RuntimeWarning: overflow encountered in exp
        return 0
    
    sigma = 1 / fx

    # if der:
    #     try:
    #         num = np.exp(-input)
    #     except OverflowError:
    #         return float('inf')
    #     except RuntimeWarning:
    #         # RuntimeWarning: overflow encountered in exp
    #         return float('inf')

    #     try:
    #         den = math.pow(fx, 2)
    #     except OverflowError:
    #         return 0

    #     return num / den

    # Si puo' semplificare e utilizzare solo il valore di attivazione:
    if der:
        return sigma * (1 - sigma)
        
    # return 1 / fx
    return sigma

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

    clipped_input = np.clip(input, -709, 709)  # Clipping per evitare overflow
    
    try:
        sinh = (np.exp(clipped_input) - np.exp(-clipped_input)) / 2
    except OverflowError:
        return float('inf')
    except RuntimeWarning:
        # RuntimeWarning: overflow encountered in exp
        return float('inf')
    
    try:
        cosh = (np.exp(clipped_input) + np.exp(-clipped_input)) / 2
    except OverflowError:
        return 0
    except RuntimeWarning:
        # RuntimeWarning: overflow encountered in exp
        return 0

    if der:
        try:
            return 1 / (cosh ** 2)
        except OverflowError:
            return 0
        except RuntimeWarning:
            # RuntimeWarning: overflow encountered in scalar power
            return 0
    
    return sinh / cosh
    # RuntimeWarning: invalid value encountered in scalar divide

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
        -   prediction: e' l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: e' l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto al target.

        Returns:
        -   se der=False, restituisce la somma dei quadrati degli errori componente per componente.
        -   se der=True, invece, restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto al target.
    """

    # print(prediction)
    # print(target)

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
        # Nel calcolare la derivata prima, il prodotto (1/2) * 2 si cancella.
        # Per calcolare la matrice jacobiana, restituiamo l'intero vettore.
        # return np.linalg.norm(errors)
        return errors
    
    # return (np.linalg.norm(errors) ** 2) / 2
    return (errors ** 2) / 2

# end

def softmax(prediction : np.ndarray) -> np.ndarray:
    """
        ...

        Parameters:
        -   

        Returns:
        -   
    """

    try:
        # y_exp = np.exp(prediction - prediction.max(axis=0))
        prediction -= np.max(prediction, axis=-1, keepdims=True)
        y_exp = np.exp(prediction)
    except OverflowError:
        # La divisione per il denominatore che tende a 'inf' e' pari a 0.
        return float('inf')

    # return y_exp / np.sum(y_exp, axis=0)
    return y_exp / np.sum(y_exp, axis=-1, keepdims=True)

# end

def cross_entropy(
        prediction : np.ndarray,
        target : np.ndarray,
        der : bool = False
) -> float | np.ndarray:
    
    """
        E' una funzione di errore tipicamente utilizzata per i problemi di classificazione.

        Parameters:
        -   prediction: e' l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: e' l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto al target.

        Returns:
        -   se der=False, ...
        -   se der=True, invece, ne restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto alla predizione.
    """

    if der:
        num = prediction - target
        den = prediction * (1 - prediction)
        return num / den
    
    # Applica il logaritmo solo alle componenti maggiori di 0.0.
    return -np.sum(target * np.log(prediction, where=prediction > 0.0))

# end

def cross_entropy_softmax(
        prediction : np.ndarray,
        target : np.ndarray,
        der : bool = False
) -> float | np.ndarray:
    
    """
        E' una funzione di errore tipicamente utilizzata per i problemi di classificazione.

        Parameters:
        -   prediction: e' l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: e' l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto alla predizione.

        Returns:
        -   se der=False, ...
        -   se der=True, invece, ne restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto alla predizione.
    """

    probabilities = softmax(prediction)

    if der:
        # Questo e' il risultato della derivata parziale della cross_entropy_softmax rispetto alla predizione in input
        # return prediction - target
        return probabilities - target
    
    # Applica il logaritmo solo alle componenti maggiori di 0.0.
    return -np.sum(target * np.log(probabilities, where=probabilities > 0.0))

# end

# ########################################################################### #
# ALTRE FUNZIONI

# def stampa_matrice(matrice):
#     for riga in matrice:
#         for elemento in riga:
#             print(int(elemento), end=" ")
#         print()

def print_progress_bar(
        iteration : int,
        total : int,
        prefix : str = '',
        suffix : str = '',
        length : int = 50,
        fill : str = '#',
) -> None:
    
    """
        Genera una barra di caricamento che si aggiorna ad ogni chiamata di un loop mostrando il numero dell'iterazione corrente rispetto al totale delle iterazioni.

        Parameters:
        -   iteration: l'indice dell'iterazione corrente.
        -   total: il numero totale di iterazioni.
        -   prefix: una stringa da stampare prima della barra di caricamento.
        -   suffix: una stringa da stampare dopo la barra di caricamento.
        -   length: la lunghezza della barra di caricamento.
        -   fill: il carattere di riempimento della barra.

        Returns:
        -   None.
    """

    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {iteration} / {total} {suffix}', end='\r')

    # Stampa una nuova linea quando tutte le iterazioni sono terminate
    if iteration == total: print()

# end

# ########################################################################### #
# RIFERIMENTI

# https://www.matematika.it/public/allegati/33/Grafici_domini_derivate_funzioni_iperboliche_1_1.pdf
# https://www.matematika.it/public/allegati/33/11_44_Derivate_2_3.pdf
# https://www.quora.com/Why-is-the-sum-of-squares-in-linear-regression-divided-by-2n-where-n-is-the-number-of-observations-instead-of-just-n
# https://stats.stackexchange.com/questions/490355/why-are-there-different-formulas-for-the-quadratic-cost-function
# https://www.youtube.com/watch?v=f50tlks5caI
# https://www.digitalocean.com/community/tutorials/relu-function-in-python
# https://stackoverflow.com/questions/4050907/python-overflowerror-math-range-error
# https://www.v7labs.com/blog/cross-entropy-loss-guide