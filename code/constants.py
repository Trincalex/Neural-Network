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
    "Cifra 9",
]
""" Enumerazione delle etichette rappresentative delle classi di output del training set e del test set del MNIST dataset. """

NUMERO_CLASSI = len(ETICHETTE_CLASSI)
""" Numero di classi di output del training set e del test set del MNIST dataset. """

COPPIE_TRAINING = 2000
""" Numero di elementi da estrarre dal training set su cui eseguire la fase di addestramento della rete neurale. """

COPPIE_TEST = 2500
""" Numero di elementi da estrarre del test set su cui eseguire la fase di test della rete neurale. """

# ##### #

DEFAULT_RANDOM_SEED = 0
""" Seed di default per la generazione random di pesi e bias, utile per la riproducibilità delle fasi di addestramento e test della rete neurale. """

DEFAULT_STANDARD_DEVIATION = 1.0
""" Valore di default per la deviazione standard utilizzata per l'inizializzazione dei pesi e bias della rete neurale. """

DEFAULT_INPUT_LAYER_NEURONS = 784
""" Valore di default per il numero di neuroni dell'input layer della rete neurale. """

DEFAULT_HIDDEN_LAYER_NEURONS = [64, 10]
""" Valori di default per il numero di neuroni dei due layer interni della rete neurale. """

DEFAULT_EPOCHS = 500
""" Valore di default del numero di epoche per la fase di addestramento della rete neurale. Un'epoca e' un'esecuzione completa dell'addestramento (e validazione, se presente) sul training set (e validation set). """

DEFAULT_MINI_BATCH_SIZE = 200
""" Dimensione di default del mini-batch utilizzata durante la fase di addestramento della rete neurale. """

DEFAULT_EARLY_STOPPING_PATIENCE = 15
""" Valore di default per il numero di epoche dopo il quale fermare l'addestramento se l'errore di validazione non e' diminuito di una certa soglia. """

DEFAULT_EARLY_STOPPING_DELTA = 0.1
""" Valore di default per la soglia che l'errore di validazione deve superare entro un certo numero di epoche per capire se ci sono stati miglioramenti significativinell'addestramento. Si sceglie di default il valore 0.1 perche', essendo il dataset MNIST grande e relativamente pulito, il modello dovrebbe migliorare in modo più consistente. """

DEFAULT_LEARNING_RATE = 0.2
""" Valore di default per il tasso di apprendimento utilizzato nella fase di addestramento della rete neurale. Indica quanto i pesi debbano essere modificati in risposta all'errore calcolato. """

# ##### #

DEFAULT_LEAKY_RELU_ALPHA = 0.01
""" Valore di default per l'iperparametro "alpha", usato nella funzione di attivazione 'leaky_relu'. Determina la pendenza per i valori negativi dell'input (invece di annullarli come nella ReLU standard). Serve a migliorare la stabilita' del modello in presenza di input che possono produrre valori negativi. """

# ##### #

DEFAULT_BACK_PROPAGATION_MODE = True
""" Valore di default per la scelta dell'algoritmo di retropropagazione da utilizzare durante la fase di addestramento della rete neurale. Il valore 'True' indica l'utilizzo dell'algoritmo di '__resilient_back_propagation', mentre il valore 'False' indica l'utilizzo dell'algoritmo di '__back_propagation' e '__gradient_descent'. """

DEFAULT_RPROP_ETA_MINUS = 0.5
""" Valore di default per l'iperparametro 'eta_minus', usato nell'algoritmo di '__resilient_back_propagation'. E' un fattore di riduzione utilizzato per ridurre il passo di aggiornamento dei pesi (step size) quando il gradiente cambia segno. Serve per stabilizzare il processo di ottimizzazione. """

DEFAULT_RPROP_ETA_PLUS = 1.2
""" Valore di default per l'iperparametro 'eta_plus', usato nell'algoritmo di '__resilient_back_propagation'. E' un fattore di incremento utilizzato per aumentare il passo di aggiornamento dei pesi (step size) quando il gradiente mantiene lo stesso segno. Serve per accelerare la convergenza verso il minimo della funzione di costo. """

DEFAULT_RPROP_DELTA_MIN = 1e-6
""" Valore di default per l'iperparametro 'delta_min', usato nell'algoritmo di '__resilient_back_propagation'. Definisce il limite inferiore per il passo di aggiornamento dei pesi (step size). Serve a garantire che l'ottimizzazione con RPROP rimanga efficiente, evitando che i passi di aggiornamento diventino troppo piccoli per contribuire significativamente al processo di apprendimento. """

DEFAULT_RPROP_DELTA_MAX = 50.0
""" Valore di default per l'iperparametro 'eta_max', usato nell'algoritmo di '__resilient_back_propagation'. Definisce il limite superiore per il passo di aggiornamento dei pesi (step size). Serve a controllare la crescita del passo di aggiornamento nell'algoritmo RPROP, bilanciando l'accelerazione della convergenza al fine di mantenere un addestramento efficiente e stabile del modello. """

# ##### #

DEFAULT_RANDOM_COMBINATIONS = 5
""" Valore di default del numero di combinazioni di iperparametri da testare nell'utilizzare la tecnica del random search. """

DEFAULT_K_FOLD_VALUE = 10
""" Valore di default del numero di fold in cui dividere il training set per il tuning degli iperparametri tramite utilizzo della tecnica della k-fold cross validation. """

# ##### #

PIXEL_INTENSITY_LEVELS = 255
""" Il numero di livelli di intensita' della scala di grigio del singolo pixel nelle immagini grayscale a 8-bit del MNIST dataset. """

DIMENSIONE_IMMAGINE = 28
""" Indica il numero di pixel su una singola dimensione delle immagini del MNIST dataset. Siccome le immagini di questo dataset sono quadrate, il numero di pixel sulle due dimensione e' lo stesso. """

# ##### #

PRINT_DATE_TIME_FORMAT = "%d-%m-%Y, %H:%M:%S"
""" Formato di default per la visualizzazione di data e ora nelle stampe in console. """

OUTPUT_DATE_TIME_FORMAT = "%Y-%m-%d_%H-%M"
""" Formato di default per la visualizzazione di data e ora nei file di output. """

# ##### #

OUTPUT_DIRECTORY = "../output/"
""" Directory dove salvare i file di output. """

DEBUG_MODE = False
""" Consente di attivare (con il valore 'True') la modalita' di debug, per stampare in console i valori attuali delle strutture dati coinvolte nella fase di addestramento della rete neurale. """

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
    Un'enumerazione che raccoglie le modalità di visualizzazione dei risultati di testing.
    -   NONE : per stampare i risultati del testing direttamente in console, senza creare e/o salvare alcun grafico.

    -   REPORT : per stampare un report generale, cioe' un barchart che mostra a quale categoria appartengono le predizioni restituite in output dalla rete neurale.

    -   HIGH_CONFIDENCE_CORRECT : oltre a stampare il report generale, fornisce anche i grafici delle sole predizioni corrette ad alta confidenza.
    -   LOW_CONFIDENCE_CORRECT : oltre a stampare il report generale, fornisce anche i grafici delle sole predizioni corrette a bassa confidenza.
    -   ALMOST_CORRECT : oltre a stampare il report generale, fornisce anche i grafici delle sole predizioni errate che superano la soglia di confidenza sull'etichetta esatta.
    -   WRONG : oltre a stampare il report generale, fornisce anche i grafici di tutte le altre predizioni errate.
    -   ALL : oltre a stampare il report generale, fornisce anche i grafici di tutte le singole predizioni divisi nelle quattro categorie di cui sopra (in sottocartelle).
    
"""

PLOT_SEARCH_FIGSIZE = (22, 12)
""" Indica le dimensioni di altezza e larghezza del report della singola predizione. """

PLOT_TESTING_FIGSIZE = (12, 4)
""" Indica le dimensioni di altezza e larghezza del report della singola predizione. """

PLOT_TESTING_IMAGE_PLOT_INDEX = 0
""" È l'indice di colonna dell'immagine in scala di grigi della cifra scritta a mano contenuta nel test set nel report della singola predizione. """

PLOT_TESTING_BAR_CHART_INDEX = 1
""" È l'indice di colonna del barchart della distribuzione di probabilita' della predizione in output nel report della singola predizione. """

PLOT_TESTING_CONFIDENCE_THRESHOLD = 0.55
""" Indica la soglia di confidenza che la predizione in output deve superare affinche' possa essere categorizzata come "risultato corretto ad alta confidenza". Il suo complemento a 1 (cioe' 0.45), e', invece, utilizzato per categorizzare le predizioni in output come "risultati quasi corretti". """

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