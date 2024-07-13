"""

    constants.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene le definizioni dei valori costanti, utili per la definizione di alcuni parametri nel codice, e dei codici di errore, utili invece a rappresentare le cause di terminazione del programma.

"""

# ########################################################################### #
# LIBRERIE

import numpy as np
from typing import Callable, Optional
from enum import Enum

# ########################################################################### #
# CLASSI DI ECCEZIONE

class LayerError(Exception):
    """ Eccezione lanciata da errori relativi ai layer della rete neurale. """

    def __init__(self, value):
        self.value = value
    # end
 
    def __str__(self):
        return(repr(self.value))
    # end

# end class LayerError

class InputLayerError(Exception):
    """ Eccezione lanciata da errori relativi all'input layer della rete neurale. """
 
    def __init__(self, value):
        self.value = value
    # end
 
    def __str__(self):
        return(repr(self.value))
    # end

# end class InputLayerError

class TrainError(Exception):
    """ Eccezione lanciata da errori relativi alla fase di addestramento. """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class TrainError

class TestError(Exception):
    """ Eccezione lanciata da errori relativi alla fase di test. """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class TestError

class ActivationFunctionError(Exception):
    """ Eccezione lanciata da errori relativi all'esecuzione della funzione di attivazione di un layer della rete neurale. """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class ActivationFunctionError

class ErrorFunctionError(Exception):
    """ Eccezione lanciata da errori relativi all'esecuzione della funzione di errore della rete neurale. """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class ErrorFunctionError

# ########################################################################### #
# COSTANTI e ENUMERAZIONI

ETICHETTE_CLASSI = [
    "Cifra 0",
    "Cifra 1",
    "Cifra 2",
    "Cifra 3",
    "Cifra 4",
    "Cifra 5",
    "Cifra 6",
    "Cifra 7",
    "Cifra 8",
    "Cifra 9"
]
""" La lista delle etichette rappresentative delle classi di output del training set e del test set del MNIST dataset. """

NUMERO_CLASSI = len(ETICHETTE_CLASSI)
""" Numero di classi di output del training set e del test set del MNIST dataset. Il suo valore e' calcolato al tempo di esecuzione in base al numero di etichette disponibili. """

COPPIE_TRAINING = 12500
""" Numero di elementi da estrarre dal training set su cui eseguire la fase di addestramento della rete neurale. """

COPPIE_TEST = 2500
""" Numero di elementi da estrarre del test set su cui eseguire la fase di test della rete neurale. """

# ##### #

DEFAULT_RANDOM_SEED = 0
""" Valore di default del seed per la generazione random di pesi e bias, utile per la riproducibilita' delle fasi di addestramento e test della rete neurale. """

DEFAULT_DISTRIBUTION_MEAN = 0.0
""" Valore di default per la media della distribuzione gaussiana utilizzata per l'inizializzazione dei pesi e bias della rete neurale. Questo valore e' stato scelto per bilanciare le attivazioni positive e negative e, di conseguenza, supportare una convergenza più rapida e stabile durante la fase di addestramento. """

DEFAULT_STANDARD_DEVIATION = 1.0
""" Valore di default per la deviazione standard utilizzata per l'inizializzazione dei pesi e bias della rete neurale. Questo valore e' stato scelto per evitare che i pesi iniziali siano troppo grandi o troppo piccoli, riducendo il rischio di saturazione dei neuroni. """

DEFAULT_INPUT_LAYER_NEURONS = 784
""" Valore di default per il numero di neuroni dell'input layer della rete neurale. Questo valore e' dato dalla dimensione di una singola immagine del MNIST dataset in input alla rete neurale (e.g. 28x28). """

DEFAULT_LAYER_NEURONS = [64, NUMERO_CLASSI]
""" Valori di default per il numero di neuroni dell'unico layer interno e dell'output layer della rete neurale. Il numero di neuroni del layer interno e' stato scelto per l'alta dimensionalita' delle immagini nel MNIST dataset, mentre il numero di neuroni per l'output layer e' stato scelto in base al numero di classi in cui sono suddivise le immagini del MNIST dataset. """

DEFAULT_EPOCHS = 500
""" Valore di default del numero di epoche per la fase di addestramento della rete neurale. Un'epoca e' un'esecuzione completa dell'addestramento (e validazione, se presente) sul training set (e validation set). """

DEFAULT_MINI_BATCH_SIZE = 1125
""" Dimensione di default del mini-batch utilizzata durante la fase di addestramento della rete neurale. Questo valore e' stato scelto per avere esattamente 10 mini-batch di addestramento. """

DEFAULT_EARLY_STOPPING_PATIENCE = 15
""" Valore di default per il numero di epoche dopo il quale fermare l'addestramento se l'errore di validazione non diminuisce di una certa soglia (e.g. se si e' raggiunta la convergenza oppure se l'errore di validazione ricomincia a salire). """

DEFAULT_EARLY_STOPPING_DELTA = 0.1
""" Valore di default per la soglia che l'errore di validazione deve superare affinche' si possa dire che la configurazione attuale di pesi e bias porta un miglioramento significativo nell'addestramento. E' stato scelto il valore 0.1 perche', essendo il MNIST dataset grande e relativamente pulito, il modello dovrebbe migliorare in modo piu' consistente. """

DEFAULT_LEARNING_RATE = 0.2
""" Valore di default per il tasso di apprendimento utilizzato nella fase di addestramento della rete neurale. Indica quanto i pesi debbano essere modificati in risposta all'errore calcolato. """

# ##### #

DEFAULT_LEAKY_RELU_ALPHA = 0.01
""" Valore di default per l'iper-parametro 'alpha', usato nella funzione di attivazione 'leaky_relu'. Determina la pendenza per i valori negativi dell'input (invece di annullarli come nella ReLU standard). Serve a migliorare la stabilita' del modello in presenza di input che possono produrre attivazioni negative. """

# ##### #

DEFAULT_BACK_PROPAGATION_MODE = True
""" Valore di default per la scelta dell'algoritmo di retropropagazione da utilizzare durante la fase di addestramento della rete neurale. Il valore 'True' indica l'utilizzo dell'algoritmo di '__resilient_back_propagation', mentre il valore 'False' indica l'utilizzo degli algoritmi di '__back_propagation' e '__gradient_descent'. """

DEFAULT_RPROP_ETA_MINUS = 0.5
""" Valore di default per l'iper-parametro 'eta_minus', usato nell'algoritmo di '__resilient_back_propagation'. E' il fattore di riduzione utilizzato per ridurre il passo di aggiornamento dei pesi (step size) quando il gradiente cambia segno. Serve per stabilizzare il processo di ottimizzazione. """

DEFAULT_RPROP_ETA_PLUS = 1.2
""" Valore di default per l'iper-parametro 'eta_plus', usato nell'algoritmo di '__resilient_back_propagation'. E' il fattore di incremento utilizzato per aumentare il passo di aggiornamento dei pesi (step size) quando il gradiente mantiene lo stesso segno. Serve per accelerare la convergenza verso il minimo della funzione di costo. """

DEFAULT_RPROP_DELTA_MIN = 1e-6
""" Valore di default per l'iper-parametro 'delta_min', usato nell'algoritmo di '__resilient_back_propagation'. Definisce il limite inferiore per il passo di aggiornamento dei pesi (step size). Serve a garantire che l'ottimizzazione con RPROP rimanga efficiente, evitando che i passi di aggiornamento diventino troppo piccoli per contribuire significativamente al processo di apprendimento. """

DEFAULT_RPROP_DELTA_MAX = 50.0
""" Valore di default per l'iper-parametro 'eta_max', usato nell'algoritmo di '__resilient_back_propagation'. Definisce il limite superiore per il passo di aggiornamento dei pesi (step size). Serve a controllare la crescita del passo di aggiornamento nell'algoritmo RPROP, bilanciando l'accelerazione della convergenza al fine di mantenere un processo di addestramento del modello efficiente e stabile. """

# ##### #

DEFAULT_RANDOM_COMBINATIONS = 5
""" Valore di default del numero di combinazioni di iper-parametri da testare nell'utilizzare la tecnica del random search. """

DEFAULT_K_FOLD_VALUE = 10
""" Valore di default del numero di fold in cui dividere il training set per il tuning degli iper-parametri tramite utilizzo della tecnica della k-fold cross validation. """

# ##### #

PIXEL_INTENSITY_LEVELS = 255
""" Il numero di livelli di intensita' della scala di grigio del singolo pixel nelle immagini grayscale a 8-bit del MNIST dataset. """

DIMENSIONE_IMMAGINE = 28
""" Il numero di pixel su una singola dimensione delle immagini del MNIST dataset. Siccome le immagini di questo dataset sono quadrate, il numero di pixel sulle due dimensioni e' lo stesso. """

# ##### #

DEBUG_MODE = False
""" Consente di attivare (con il valore 'True') la modalita' di debug, per stampare in console i valori attuali delle strutture dati coinvolte nella fase di addestramento della rete neurale. """

PRINT_DATE_TIME_FORMAT = "%d-%m-%Y, %H:%M:%S"
""" Formato di default per la visualizzazione di data e ora nelle stampe in console. """

OUTPUT_DATE_TIME_FORMAT = "%Y-%m-%d_%H-%M"
""" Formato di default per la visualizzazione di data e ora nelle directory di output. """

OUTPUT_DIRECTORY = "../output/"
""" Percorso relativo della directory dove salvare tutti i file di output, tra cui le configurazioni di parametri delle reti addestrate e i report (su immagine e su file '.csv') del tuning degli iper-parametri tramite Grid Search e Random Search. """

# ##### #

PlotTestingMode = Enum('PlotTestingMode', [
    'NONE',
    'REPORT',
    'HIGH_CONFIDENCE_CORRECT',
    'LOW_CONFIDENCE_CORRECT',
    'ALMOST_CORRECT',
    'WRONG',
    'ALL'
])
"""
    Un'enumerazione che raccoglie le modalita' di visualizzazione dei risultati di testing.
    -   'NONE' : per stampare i risultati del testing direttamente in console, senza creare e/o salvare alcun grafico.

    -   'REPORT' : per stampare un report generale, cioe' un barchart che mostra a quale categoria appartengono le predizioni restituite in output dalla rete neurale.

    -   'HIGH_CONFIDENCE_CORRECT' : oltre a stampare il report generale, fornisce anche i grafici delle sole predizioni corrette ad alta confidenza.
    -   'LOW_CONFIDENCE_CORRECT' : oltre a stampare il report generale, fornisce anche i grafici delle sole predizioni corrette a bassa confidenza.
    -   'ALMOST_CORRECT' : oltre a stampare il report generale, fornisce anche i grafici delle sole predizioni errate che superano la soglia di confidenza sull'etichetta esatta.
    -   'WRONG' : oltre a stampare il report generale, fornisce anche i grafici di tutte le altre predizioni errate.
    -   'ALL' : oltre a stampare il report generale, fornisce anche i grafici di tutte le singole predizioni divisi nelle quattro categorie di cui sopra (in sottocartelle).
    
"""

PLOT_SEARCH_FIGSIZE = (22, 12)
""" Le dimensioni di altezza e larghezza del report del tuning degli iper-parametri tramite Grid Search e Random Search. """

PLOT_TESTING_FIGSIZE = (12, 4)
""" Le dimensioni di altezza e larghezza del report della singola predizione. """

PLOT_TESTING_IMAGE_PLOT_INDEX = 0
""" L'indice di colonna nel report della singola predizione in cui disegnare l'immagine in scala di grigi della cifra scritta a mano contenuta nel test set. """

PLOT_TESTING_BAR_CHART_INDEX = 1
""" L'indice di colonna nel report della singola predizione in cui disegnare il barchart della distribuzione di probabilita' della predizione in output. """

PLOT_TESTING_CONFIDENCE_THRESHOLD = 0.55
""" La soglia di confidenza che la predizione in output deve superare affinche' possa essere categorizzata come "risultato corretto ad alta confidenza". Il suo complemento a 1 (e.g. 0.45) e', invece, utilizzato per categorizzare le predizioni in output come "risultati quasi corretti". """

# ########################################################################### #
# ALIAS DI TIPO

ActivationFunctionType = Callable[[float | np.ndarray, Optional[bool]], float | np.ndarray]
"""
    Semplifica in un alias di tipo la struttura della firma di una funzione di attivazione.
    Accetta in input: un valore 'float' oppure un 'np.ndarray'; opzionalmente, un valore 'bool'. Restituisce in output: un valore 'float' oppure un 'np.ndarray'.
"""

ErrorFunctionType = Callable[[np.ndarray, np.ndarray, Optional[bool]], float | np.ndarray]
"""
    Semplifica in un alias di tipo la struttura della firma di una funzione di errore.
    Accetta in input: due 'np.ndarray'; opzionalmente, un valore 'bool'.
    Restituisce in output: un valore 'float' oppure un 'np.ndarray'.
"""

# ########################################################################### #
# RIFERIMENTI

# https://www.geeksforgeeks.org/user-defined-exceptions-python-examples/
# https://stackoverflow.com/questions/6060635/convert-enum-to-int-in-python
# https://keras.io/api/callbacks/early_stopping/