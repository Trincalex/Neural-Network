"""

    training_params.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    ...

"""

# ########################################################################### #
# LIBRERIE

import constants
import numpy as np

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE TrainingParams

class TrainingParams:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def batch_size(self) -> int:
        """ ... """
        return self._batch_size
    # end

    @property
    def epochs(self) -> int:
        """ ... """
        return self._epochs
    # end

    @property
    def learning_rate(self) -> float:
        """ ... """
        return self._learning_rate
    # end

    @property
    def rprop(self) -> bool:
        """ ... """
        return self._rprop
    # end

    @property
    def eta_minus(self) -> float:
        """ ... """
        return self._eta_minus
    # end

    @property
    def eta_plus(self) -> float:
        """ ... """
        return self._eta_plus
    # end

    @property
    def delta_min(self) -> float:
        """ ... """
        return self._delta_min
    # end

    @property
    def delta_max(self) -> float:
        """ ... """
        return self._delta_max
    # end

    @property
    def es_patience(self) -> int:
        """ ... """
        return self._es_patience
    # end

    @property
    def es_delta(self) -> float:
        """ ... """
        return self._es_delta
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            batch_size : int = constants.DEFAULT_MINI_BATCH_SIZE,
            epochs : int = constants.DEFAULT_EPOCHS,
            learning_rate : float = constants.DEFAULT_LEARNING_RATE,
            rprop : bool = constants.DEFAULT_BACK_PROPAGATION_MODE,
            eta_minus : float = constants.DEFAULT_RPROP_ETA_MINUS,
            eta_plus : float = constants.DEFAULT_RPROP_ETA_PLUS,
            delta_min : float = constants.DEFAULT_RPROP_DELTA_MIN,
            delta_max : float = constants.DEFAULT_RPROP_DELTA_MAX,
            es_patience : int = constants.DEFAULT_EARLY_STOPPING_PATIENCE,
            es_delta : float = constants.DEFAULT_EARLY_STOPPING_DELTA
    ) -> None:
        
        """
            E' il costruttore della classe TrainingParams.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   batch_size : ...
            -   epochs : il numero di iterazioni per cui il modello deve essere addestrato. Un'epoca e' un'esecuzione completa dell'addestramento attraverso l'intero training_set.
            -   learning_rate : e' un parametro utilizzato per l'aggiornamento dei pesi che indica quanto i pesi debbano essere modificati in risposta all'errore calcolato.
            -   rprop : ...
            -   eta_minus : ...
            -   eta_plus : ...
            -   delta_min : ...
            -   delta_max : ...
            -   es_patience : ...
            -   es_delta : ...

            Returns:
            -   None.
        """

        if not batch_size > 0:
            raise constants.TrainError(f"Il valore di 'batch_size' ({batch_size}) deve essere maggiore di 0.")
        self._batch_size = int(batch_size)
        
        if not epochs > 0:
            raise constants.TrainError(f"Il valore di 'epochs' ({epochs}) deve essere maggiore di 0.")
        self._epochs = int(epochs)

        if not (learning_rate >= 0 and learning_rate <= 1):
            raise constants.TrainError(f"Il valore di 'learning_rate' ({learning_rate}) deve essere compreso tra 0 e 1.")
        self._learning_rate = float(learning_rate)
        
        self._rprop = bool(rprop)
        
        if not (eta_minus > 0 and eta_minus < 1):
            raise constants.TrainError(f"Il valore di 'eta_minus' ({eta_minus}) deve essere compreso tra 0 e 1.")
        self._eta_minus = float(eta_minus)

        if not eta_plus > 1:
            raise constants.TrainError(f"Il valore di 'eta_plus' ({eta_plus}) deve essere maggiore di 1.")
        self._eta_plus = float(eta_plus)
        
        if not delta_min > 0:
            raise constants.TrainError(f"Il valore di 'delta_min' ({delta_min}) deve essere maggiore di 0.")
        self._delta_min = float(delta_min)
        
        if not delta_max > 0:
            raise constants.TrainError(f"Il valore di 'delta_max' ({delta_max}) deve essere maggiore di 0.")
        self._delta_max = float(delta_max)
        
        if not es_patience > 0:
            raise constants.TrainError(f"Il valore di 'es_patience' ({es_patience}) deve essere maggiore di 0.")
        self._es_patience = int(es_patience)

        if not es_delta > 0:
            raise constants.TrainError(f"Il valore di 'es_delta' ({es_delta}) deve essere maggiore di 0.")
        self._es_delta = float(es_delta)
    
    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe TrainingParams.

            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'TrainingParams(\n\tbatch_size = {self.batch_size},\n\t epochs = {self.epochs},\n\t learning_rate = {self.learning_rate},\n\t rprop = {self.rprop},\n\t eta_minus = {self.eta_minus},\n\t eta_plus = {self.eta_plus},\n\t delta_min = {self.delta_min},\n\t delta_max = {self.delta_max},\n\t es_patience = {self.es_patience},\n\t es_delta = {self.es_delta}\n)'
    
    # end

# end class TrainingParams

# ########################################################################### #
# RIFERIMENTI

