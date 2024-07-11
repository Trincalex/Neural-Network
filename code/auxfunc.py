"""

    auxfunc.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene alcune funzionalita' aggiuntive per la creazione della rete neurale e l'esecuzione del programma, tra cui la definizione delle funzioni di attivazione e delle funzioni di errore.

"""

# ########################################################################### #
# LIBRERIE

from training_report import TrainingReport
import constants

import numpy as np

# ########################################################################### #
# FUNZIONI DI ATTIVAZIONE

def leaky_relu(input : float | np.ndarray, der : bool = False) -> float | np.ndarray:
    """
        Calcola il valore di attivazione di uno o piu' neuroni utilizzando un miglioramento della classica ReLU (Rectified Linear Unit), in quanto aggiunge una pendenza costante (specificata dall'iper-parametro 'alpha') per gli input negativi, garantendo una maggiore stabilità nell’apprendimento dei modelli.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'attivazione della Leaky ReLU.
        -   se der=True, invece, ne restituisce la derivata prima.
    """

    if der:
        return np.where(input >= 0, 1, constants.DEFAULT_LEAKY_RELU_ALPHA)

    return np.where(input >= 0, input, constants.DEFAULT_LEAKY_RELU_ALPHA * input)

# end

def sigmoid(input : float | np.ndarray, der : bool = False) -> float | np.ndarray:
    """
        Calcola il valore di attivazione di uno o piu' neuroni utilizzando la funzione logistica per normalizzare l'input nell’intervallo tra 0 e 1.

        Parameters:
        -   input : il valore o la matrice di valori di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'attivazione della sigmoide.
        -   se der=True, invece, ne restituisce la derivata prima.
    """

    # Il piu' grande valore rappresentabile da un numpy float 1.797e+308, il cui logaritmo e' circa 709.
    # Si esegue il clipping per evitare problemi di overflow.
    clipped_input = np.clip(input, -709, 709)

    try:
        fx = 1 + np.exp(-clipped_input)
    except (OverflowError, RuntimeWarning):
        # RuntimeWarning: overflow encountered in exp
        return 0
    
    sigma = 1 / fx

    # Si puo' semplificare e utilizzare solo il valore di attivazione:
    if der:
        return sigma * (1 - sigma)
        
    # return 1 / fx
    return sigma

# end

def tanh(input : float | np.ndarray, der : bool = False) -> float | np.ndarray:
    """
        Calcola il valore di attivazione di uno o piu' neuroni utilizzando la tangente iperbolica per normalizzare l'input nell’intervallo tra -1 e 1.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'attivazione della tangente iperbolica.
        -   se der=True, invece, ne restituisce la derivata prima.
    """

    # Il piu' grande valore rappresentabile da un numpy float 1.797e+308, il cui logaritmo e' circa 709.
    # Si esegue il clipping per evitare problemi di overflow.
    clipped_input = np.clip(input, -709, 709)
    
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
            ret = 1 / (cosh ** 2)
            return ret
        except OverflowError:
            return 0
        except RuntimeWarning:
            # RuntimeWarning: overflow encountered in scalar power
            return 0
    
    return sinh / cosh
    # RuntimeWarning: invalid value encountered in scalar divide

# end
    
def identity(input : float | np.ndarray, der : bool = False) -> float | np.ndarray:
    """
        Calcola il valore di attivazione di uno o piu' neuroni utilizzando la funzione identita', mantenendo invariato l’input.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno.

        Returns:
        -   se der=False, restituisce l'input.
        -   se der=True, invece, ne restituisce la derivata prima, cioe' 1.
    """
    
    if der:
        if isinstance(input, np.ndarray):
            return np.ones(input.shape)
        return 1.0
    
    return input

# end

def softmax(input : float | np.ndarray, der : bool = False) -> float | np.ndarray:
    """
        Converte i valori di attivazione di uno o più neuroni in un vettore di probabilità, dove ogni valore rappresenta la probabilità che l'input appartenga a una determinata classe.
        Rispetto alle altre funzioni di attivazione descritte in questa sezione, la funzione softmax è una funzione globale, perché dipende dall'input di tutti i neuroni. Per questo motivo, è utilizzata principalmente per i problemi di classificazione multi-classe.
        In particolare, in questa libreria, non è fornita l'implementazione della derivata prima della softmax che, se utilizzata, lancia l'eccezione NotImplementedError.

        Parameters:
        -   input : il valore di cui applicare la funzione di attivazione.
        -   der : indica se si vuole calcolare la derivata prima o meno

        Returns:
        -   se der=False restituisce il risultato della softmax
        -   se der=True lancia un'eccezione poichè non è possibile calcolare la derivata della softmax.
    """
    
    if der:
        raise NotImplementedError("Non e' possibile calcolare la derivata della softmax.")

    try:
        # Applichiamo una normalizzazione per evitare instabilita' nei risultati
        input -= np.max(input, axis=-1, keepdims=True)

        # Il piu' grande valore rappresentabile da un numpy float 1.797e+308, il cui logaritmo e' circa 709.
        # Si esegue il clipping per evitare problemi di overflow.
        clipped_input = np.clip(input, -709, 709)

        y_exp = np.exp(clipped_input)
    except OverflowError:
        # La divisione per il numeratore che tende a 'inf' e' pari a 0.
        return float('inf')

    # return y_exp / np.sum(y_exp, axis=0)
    return y_exp / np.sum(y_exp, axis=-1, keepdims=True)

# end

# ########################################################################### #
# FUNZIONI DI ERRORE

def sum_of_squares(
        predictions : np.ndarray,
        target : np.ndarray,
        der : bool = False
) -> float | np.ndarray:
    
    """
        E' una funzione di errore tipicamente utilizzata per i problemi di regressione.

        Parameters:
        -   predictions: e' l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: e' l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto al target.

        Returns:
        -   se der=False, restituisce la somma dei quadrati degli errori componente per componente.
        -   se der=True, invece, restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto al target.
    """

    # print(predictions)
    # print(target)

    # Calcolo delle distanze tra predizioni e target (errori)
    errors = predictions - target
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
    return np.sum((errors ** 2)) / 2

# end

def cross_entropy(
        predictions : np.ndarray,
        target : np.ndarray,
        der : bool = False
) -> float | np.ndarray:
    
    """
        E' una funzione di errore tipicamente utilizzata per i problemi di classificazione.

        Parameters:
        -   predictions: e' l'output fornito dalla rete neurale su una determinata coppia del dataset.
        -   target: e' l'etichetta di classificazione di una determinata coppia del dataset.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto al target.

        Returns:
        -   se der=False, restituisce l'entropia incrociata delle due variabili aleatorie discrete relative al target e alla predizione.
        -   se der=True, invece, ne restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto alla predizione.
    """

    if der:
        # num = predictions - target
        # den = predictions * (1 - predictions)
        # return num / den
        return predictions - target
    
    # Applica il logaritmo solo alle componenti maggiori di 0.0.
    return -np.sum(target * np.log(predictions, where=predictions > 0.0))

# end

def cross_entropy_softmax(
        predictions : np.ndarray,
        targets : np.ndarray,
        der : bool = False
) -> float | np.ndarray:
    
    """
        E' una funzione di errore tipicamente utilizzata per i problemi di classificazione.

        Parameters:
        -   predictions: sono le predizioni (cioe', i valori di attivazione dell'output layer) fornite dalla rete neurale per un insieme di coppie del dataset.
        -   targets: sono le etichette di classificazione dell'insieme di coppie del dataset di cui sono state calcolate le predizioni.
        -   der: permette di distinguere se si vuole calcolare la funzione o la matrice delle derivate prime parziali rispetto alle predizioni.

        Returns:
        -   se der=False, restituisce l'entropia incrociata delle due variabili aleatorie discrete relative al target e alla predizione.
        -   se der=True, invece, ne restituisce la matrice delle derivate prime parziali (matrice jacobiana) rispetto alle predizioni.
    """

    if not isinstance(predictions, np.ndarray):
        raise constants.ErrorFunctionError("La matrice delle predizioni deve essere di tipo 'numpy.ndarray'.")
    if not isinstance(targets, np.ndarray):
        raise constants.ErrorFunctionError("La matrice delle etichette deve essere di tipo 'numpy.ndarray'.")
    if not predictions.shape == targets.shape:
        raise constants.ErrorFunctionError(f"Il numero di predizioni ({predictions.shape}) e di etichette ({targets.shape}) non sono compatibili.")

    prob_predictions = softmax(predictions)

    if der:
        # Questo e' il risultato della derivata parziale della cross_entropy_softmax rispetto alla predizione in input
        # return predictions - targets
        return prob_predictions - targets
    
    # Applica il logaritmo solo alle componenti maggiori di 0.0.
    return -np.sum(targets * np.log(prob_predictions, where=prob_predictions > 0.0))

# end

# ########################################################################### #
# ALTRE FUNZIONI

def print_progress_bar(
        iteration : int,
        total : int,
        prefix : str = '',
        suffix : str = '',
        length : int = 50,
        fill : str = '#'
) -> None:
    
    """
        Stampa in console una barra di caricamento che si aggiorna ad ogni chiamata di un loop mostrando il numero dell'iterazione corrente rispetto al totale delle iterazioni.

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

def compute_batches(
        length : int,
        batch_size : int = constants.DEFAULT_MINI_BATCH_SIZE
) -> list[tuple[int, int]]:
    
    """
        Calcola gli indici di inizio e fine dei mini-batch di un dataset sulle quali eseguire la fase di addestramento della rete neurale.

        Parameters:
        -   length : e' la lunghezza del dataset.
        -   batch_size : e' la lunghezza del singolo mini-batch.
            In particolare:
            -   se e' esattamente uguale a 'length', allora si ottiene un unico batch (e.g. batch learning).
            -   se e' minore di 'length', allora si ottiene un maggior numero di batch (e.g. mini-batch learning).
            -   se e' '1', allora si ottiene il massimo numero di batch (e.g. online learning).

        Returns:
        -   batches_indexes : la lista di indici di inizio e fine dei mini-batch del dataset.
    """

    if not batch_size > 0:
        raise ValueError(f"Il numero di esempi ({batch_size}) deve essere maggiore di 0.")

    batches_indexes = []

    num_batches = length // batch_size
    end = batch_size * num_batches

    if end < length:
        num_batches += 1
    
    for c in range(num_batches):
        start   = c * batch_size
        end     = (c+1) * batch_size

        if length < end:
            end = length

        batches_indexes.append((start,end))

    # end for c
    
    return batches_indexes

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
# https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
# https://www.v7labs.com/blog/cross-entropy-loss-guide
# https://www.redcrabmath.com/Calculator/Softmax
# https://www.lokad.com/it/definizione-entropia-incrociata/