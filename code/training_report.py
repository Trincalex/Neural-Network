"""

    training_report.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione della classe TrainingReport, il cui scopo e' quello di memorizzare tutte le misure riguardanti l'addestramento di una rete neurale feed-forward fully-connected.

"""

# ########################################################################### #
# LIBRERIE

import constants
import auxfunc
import numpy as np

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE TrainingReport

class TrainingReport:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def num_epochs(self) -> float:
        """..."""
        return self._num_epochs
    # end

    @property
    def elapsed_time(self) -> float:
        """..."""
        return self._elapsed_time
    # end

    @property
    def training_error(self) -> float:
        """..."""
        return self._training_error
    # end

    @property
    def validation_error(self) -> float:
        """..."""
        return self._validation_error
    # end

    @property
    def training_accuracy(self) -> float:
        """..."""
        return self._training_accuracy
    # end

    @property
    def validation_accuracy(self) -> float:
        """..."""
        return self._validation_accuracy
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            epochs : int = 0,
            time : float = 0.0,
            t_error : float = 0.0,
            v_error : float = 0.0,
            t_accuracy : float = 0.0,
            v_accuracy : float = 0.0
    ):
        """
            E' il costruttore della classe TrainingReport.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   ... : ...

            Returns:
            -   None.
        """

        # Inizializzazione del numero di epoche.
        self._num_epochs = epochs

        # Inizializzazione del tempo di addestramento.
        self._elapsed_time = time

        # Inizializzazione delle metriche di errore.
        self._training_error = t_error
        self._validation_error = v_error

        # Inizializzazione delle metriche di accuracy.
        self._training_accuracy = t_accuracy
        self._validation_accuracy = v_accuracy
    
    # end

    # ####################################################################### #
    # METODI PRIVATI



    # ####################################################################### #
    # METODI PUBBLICI

    def compute_error(
            self,
            predictions : np.ndarray,
            targets : np.ndarray,
            err_fun : constants.ErrorFunctionType
    ) -> float:
        
        """
            Calcola l'errore della rete neurale confrontando le previsioni con i target corrispondenti.

            Parameters:
            -   predictions : la matrice contenente tutte le previsioni della rete.
            -   targets : la matrice contenente le etichette vere corrispondenti alle previsioni (ground truth).

            Returns:
            -   float : la media delle distanze tra le predizioni della rete neurale e le etichette della ground truth.
        """

        # print(predictions.shape, targets.shape)
        if not predictions.shape == targets.shape:
            raise constants.TrainError("Il numero di predizioni e di etichette non sono compatibili.")
        
        errors = [err_fun(x, y) for x, y in zip(predictions, targets)]
        return np.mean(errors)

    # end
    
    def compute_accuracy(self, predictions : np.ndarray, targets : np.ndarray) -> float:
        """
            Calcola l'accuratezza della rete neurale confrontando le previsioni con i target corrispondenti.

            Parameters:
            -   predictions : la matrice contenente tutte le previsioni della rete.
            -   targets : la matrice contenente le etichette vere corrispondenti alle previsioni (ground truth).

            Returns:
            -   float : il rapporto tra predizioni corrette e totale delle etichette, moltiplicato per 100 per ottenere il valore percentuale.
        """

        # print(predictions.shape, targets.shape)
        if not predictions.shape == targets.shape:
            raise constants.TrainError("Il numero di predizioni e di etichette non sono compatibili.")

        matches = [int(np.argmax(x) == np.argmax(y)) for x, y in zip(predictions, targets)]
        return np.sum(matches) / len(targets) * 100

    # end

    def update(self, value) -> None:
        """
            ...

            Parameters:
            -   ... : ...

            Returns:
            -   None.
        
        """

        if (not isinstance(value, type(self))):
            raise TypeError("Impossibile aggiornare il report.")

        self._num_epochs            = value.num_epochs
        self._elapsed_time          = value.elapsed_time
        self._training_error        = value.training_error
        self._validation_error      = value.validation_error
        self._training_accuracy     = value.training_accuracy
        self._validation_accuracy   = value.validation_accuracy

    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe TrainingReport.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'TrainingReport(\n\tnum_epochs = {self.num_epochs},\n\telapsed_time = {self.elapsed_time:.3f} secondi,\n\ttraining_error = {self.training_error:.5f},\n\ttraining_accuracy = {self.training_accuracy:.2f} %,\n\tvalidation_error = {self.validation_error:.5f}\n\tvalidation_accuracy = {self.validation_accuracy:.2f} %\n)'
    
    # end

# end TrainingReport

# ########################################################################### #
# RIFERIMENTI

