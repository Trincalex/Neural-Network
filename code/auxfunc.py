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
    except OverflowError:
        print(f"OverflowError: il valore 'fx' e' troppo grande.")
        fx = float('inf')

    if der:
        try:
            num = np.exp(-input)
        except OverflowError:
            print(f"OverflowError: il valore 'num' e' troppo grande.")
            num = float('inf')

        try:
            den = math.pow(fx, 2)
        except OverflowError:
            print(f"OverflowError: il valore 'den' e' troppo grande.")
            den = float('inf')

        return num / den
    
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
    
    try:
        sinh = (np.exp(input) - np.exp(-input)) / 2
    except OverflowError:
        print(f"OverflowError: il valore 'sinh' e' troppo grande.")
        sinh = float('inf')
    
    try:
        cosh = (np.exp(input) + np.exp(-input)) / 2
    except OverflowError:
        print(f"OverflowError: il valore 'cosh' e' troppo grande.")
        cosh = float('inf')

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
        # Nel calcolare la derivata prima, il prodotto (1/2) * 2 si cancella
        # Per calcolare la matrice jacobiana, non restituiamo la somma ma l'intero vettore.
        return errors
    
    return (np.linalg.norm(errors) ** 2) / 2

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
        x_exp = np.exp(prediction - prediction.max(0))
    except OverflowError:
        print(f"OverflowError: il valore 'x_exp' e' troppo grande.")
        x_exp = float('inf')

    return x_exp / np.sum(x_exp, 0)

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
        decimals : int = 1,
        length : int = 50,
        fill : str = '#',
        print_end : str = "\r"
) -> None:
    
    """
        Genera una barra di caricamento che si aggiorna ad ogni chiamata di un loop mostrando la percentuale di caricamento attuale.

        Parameters:
        -   iteration: l'indice dell'iterazione corrente.
        -   total: il numero totale di iterazioni.
        -   prefix: una stringa da stampare prima della barra di caricamento.
        -   suffix: una stringa da stampare dopo la barra di caricamento.
        -   decimals: il numero di cifre decimali da stampare della percentuale di progresso.
        -   length: la lunghezza della barra di caricamento.
        -   fill: il carattere di riempimento della barra.
        -   print_end: il carattere da stampare al termine delle iterazioni.

        Returns:
        -   None.
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)

    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)

    # Stampa una nuova linea quando tutte le iterazioni sono terminate
    if iteration == total:
        print()

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