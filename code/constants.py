'''

    constants.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene le definizioni dei valori costanti, utili per la
    definizione di alcuni parametri nel codice, e dei codici di errore,
    utili invece a rappresentare le cause di terminazione del programma.

'''

# ########################################################################### #
# LIBRERIE

import numpy as np
from typing import Callable, Optional
from enum import Enum

# ########################################################################### #
# CODICI DI ERRORE e EXCEPTIONS

class LayerError(Exception):
 
    def __init__(self, value):
        self.value = value
    # end
 
    def __str__(self):
        return(repr(self.value))
    # end

# end class LayerError

class InputLayerError(Exception):
 
    def __init__(self, value):
        self.value = value
    # end
 
    def __str__(self):
        return(repr(self.value))
    # end

# end class InputLayerError

class HiddenLayerError(Exception):
 
    def __init__(self, value):
        self.value = value
    # end
 
    def __str__(self):
        return(repr(self.value))
    # end

# end class HiddenLayerError

class TrainError(Exception):

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class TrainError

class ActivationFunctionError(Exception):

    def __init__(self, value):
        self.value = value
    # end

    def __str__(self):
        return(repr(self.value))
    # end

# end class ActivationFunctionError

class ErrorFunctionError(Exception):

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

NUMERO_CLASSI = 10
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

COPPIE_TRAINING = 12500
""" ... """

COPPIE_TEST = 2500
""" ... """

DEFAULT_K_FOLD_VALUE = 10
""" ... """

DIMENSIONE_NEURONE_INPUT = 1
""" ... """

DIMENSIONE_PIXEL = 255
""" ... """

DIMENSIONE_IMMAGINE = 28
""" ... """

DEFAULT_EPOCHS = 100
""" ... """

DEFAULT_LEARNING_RATE = 0.1
""" ... """

DEFAULT_LEAKY_RELU_ALPHA = 0.01
""" ... """

DEFAULT_RANDOM_SEED = 0
""" ... """

DEBUG_MODE = False
""" Consente di attivare la modalita' di debug per stampare in console i valori attuali delle strutture dati coinvolte nell'addestramento della rete neurale. """

DATE_TIME_FORMAT = "%d-%m-%Y, %H:%M:%S"
""" ... """

OUTPUT_DIRECTORY = "../output/"
""" ... """

class ReportTitle(Enum):
    """ ... """

    Error = 1, "Error report"
    Accuracy = 2, "Accuracy report"

    def __int__(self):
        return self.value[0]

    def __str__(self):
        return self.value[1]

# end class ReportTitle

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