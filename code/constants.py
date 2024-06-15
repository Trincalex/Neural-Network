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
# CODICI DI ERRORE e EXCEPTIONS

class LayerError(Exception):
    """ ... """

    def __init__(self, value):
        self.value = value
    # end
 
    def __str__(self):
        return(repr(self.value))
    # end

# end class LayerError

class InputLayerError(Exception):
    """ ... """
 
    def __init__(self, value):
        self.value = value
    # end
 
    def __str__(self):
        return(repr(self.value))
    # end

# end class InputLayerError

class TrainError(Exception):
    """ ... """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class TrainError

class TestError(Exception):
    """ ... """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class TestError

class ActivationFunctionError(Exception):
    """ ... """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class ActivationFunctionError

class ErrorFunctionError(Exception):
    """ ... """

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class ErrorFunctionError

# ########################################################################### #
# COSTANTI e ENUMERAZIONI

STANDARD_DEVIATION = 1.0
""" ... """

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
""" ... """

NUMERO_CLASSI = len(ETICHETTE_CLASSI)
""" ... """

COPPIE_TRAINING = 12500
""" ... """

COPPIE_TEST = 2500
""" ... """

DEFAULT_EPOCHS = 500
""" ... """

DEFAULT_MINI_BATCH_SIZE = 64
""" ... """

DEFAULT_EARLY_STOPPING_DELTA = 0.1
""" La soglia che l'errore di validazione deve superare entro un certo numero di epoche per capire se ci sono stati miglioramenti nell'addestramento. Si sceglie di default il valore 0.1 perche', essendo il dataset MNIST grande e relativamente pulito, il modello dovrebbe migliorare in modo più consistente. """

DEFAULT_EARLY_STOPPING_PATIENCE = 20
""" Il numero di epoche dopo il quale fermare l'addestramento se l'errore di validazione non e' diminuito di una certa soglia. """

DEFAULT_LEARNING_RATE = 0.1
""" ... """

DEFAULT_BACK_PROPAGATION_MODE = True
""" ... """

DEFAULT_RPROP_ETA_MINUS = 0.5
""" ... """

DEFAULT_RPROP_ETA_PLUS = 1.2
""" ... """

DEFAULT_RPROP_DELTA_MIN = 1e-6
""" ... """

DEFAULT_RPROP_DELTA_MAX = 50.0
""" ... """

DEFAULT_K_FOLD_VALUE = 10
""" ... """

DIMENSIONE_NEURONE_INPUT = 1
""" ... """

DIMENSIONE_PIXEL = 255
""" ... """

DIMENSIONE_IMMAGINE = 28
""" ... """

DEFAULT_LEAKY_RELU_ALPHA = 0.01
""" ... """

DEFAULT_RANDOM_SEED = 0
""" ... """

DEBUG_MODE = False
""" Consente di attivare la modalita' di debug per stampare in console i valori attuali delle strutture dati coinvolte nell'addestramento della rete neurale. """

PRINT_DATE_TIME_FORMAT = "%d-%m-%Y, %H:%M:%S"
""" ... """

OUTPUT_DIRECTORY = "../output/"
""" ... """

OUTPUT_DATE_TIME_FORMAT = "%Y-%m-%d_%H-%M"
""" ... """

PlotTestingMode = Enum('PlotTestingMode', [
    'NONE',
    'REPORT',
    'HIGH_CONFIDENCE_CORRECT',
    'LOW_CONFIDENCE_CORRECT',
    'ALMOST_CORRECT',
    'WRONG',
    'ALL'
])
""" ... """

PLOT_TESTING_FIGSIZE = (12, 4)
""" ... """

PLOT_TESTING_COLUMNS = 2
""" ... """

PLOT_TESTING_IMAGE_PLOT_INDEX = 0
""" ... """

PLOT_TESTING_BAR_CHART_INDEX = 1
""" ... """

PLOT_TESTING_CONFIDENCE_THRESHOLD = 0.55
""" ... """

# ########################################################################### #
# COSTANTI

ActivationFunctionType = Callable[[float | np.ndarray, Optional[bool]], float | np.ndarray]
""" ... """

ErrorFunctionType = Callable[[np.ndarray, np.ndarray, Optional[bool]], float | np.ndarray]
""" ... """

# ########################################################################### #
# RIFERIMENTI

# https://www.geeksforgeeks.org/user-defined-exceptions-python-examples/
# https://stackoverflow.com/questions/6060635/convert-enum-to-int-in-python
# https://keras.io/api/callbacks/early_stopping/