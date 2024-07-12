"""

    training_params.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione della classe TrainingParams, il cui scopo e' quello di memorizzare una parte dei valori degli iper-parametri riguardanti l'addestramento di una rete neurale feed-forward fully-connected (perche' altri sono intrinsecamente memorizzati nell'implementazione della classe NeuralNetwork).

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
        """ Il numero di esempi di addestramento contenuti in un singolo mini-batch. """
        return self._batch_size
    # end

    @property
    def epochs(self) -> int:
        """ Il massimo numero di iterazioni per cui il modello deve essere addestrato. """
        return self._epochs
    # end

    @property
    def learning_rate(self) -> float:
        """ Il tasso di apprendimento utilizzato nella fase di addestramento per l'aggiornamento dei pesi / bias. """
        return self._learning_rate
    # end

    @property
    def rprop(self) -> bool:
        """ Un flag che indica quale algoritmo di retro-propagazione utilizzare nella fase di addestramento. """
        return self._rprop
    # end

    @property
    def eta_minus(self) -> float:
        """ Il fattore di riduzione utilizzato per ridurre il passo di aggiornamento dei pesi (step size) quando il gradiente cambia segno. Serve per stabilizzare il processo di ottimizzazione. """
        return self._eta_minus
    # end

    @property
    def eta_plus(self) -> float:
        """ Il fattore di incremento utilizzato per aumentare il passo di aggiornamento dei pesi (step size) quando il gradiente conserva lo stesso segno. Serve per accelerare la convergenza verso il minimo della funzione di costo. """
        return self._eta_plus
    # end

    @property
    def delta_min(self) -> float:
        """ Il limite inferiore che il passo di aggiornamento dei pesi (step size) puo' raggiungere, assicurando che gli aggiornamenti non diventino poco significativi. """
        return self._delta_min
    # end

    @property
    def delta_max(self) -> float:
        """ Il limite superiore che il passo di aggiornamento dei pesi (step size) puo' raggiungere durante l'addestramento, utilizzato per limitare l'entita' le oscillazioni. """
        return self._delta_max
    # end

    @property
    def es_patience(self) -> int:
        """ Il numero di epoche dopo il quale fermare il processo di addestramento se l'errore di validazione non e' diminuito di una certa soglia (e.g. si e' raggiunta la convergenza oppure l'errore di validazione ricomincia a salire). """
        return self._es_patience
    # end

    @property
    def es_delta(self) -> float:
        """ La soglia che l'errore di validazione deve superare rispetto alla miglior epoca durante la fase di addestramento per capire se ci sono stati miglioramenti significativi. """
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
            -   batch_size : e' il numero di esempi di addestramento contenuti in un singolo mini-batch.
            -   epochs : e' il massimo numero di iterazioni per cui il modello deve essere addestrato.
            -   learning_rate : e' un iper-parametro utilizzato per l'aggiornamento dei pesi.
            -   rprop : e' un flag che indica quale algoritmo di retro-propagazione utilizzare nella fase di addestramento.
            -   eta_minus : e' il fattore di riduzione utilizzato per ridurre il passo di aggiornamento dei pesi (step size) quando il gradiente cambia segno.
            -   eta_plus : e' il fattore di incremento utilizzato per aumentare il passo di aggiornamento dei pesi (step size) quando il gradiente mantiene lo stesso segno.
            -   delta_min : e' il limite inferiore per il passo di aggiornamento dei pesi (step size).
            -   delta_max : e' il limite superiore per il passo di aggiornamento dei pesi (step size).
            -   es_patience : e' il numero di epoche dopo il quale fermare l'addestramento se l'errore di validazione non e' diminuito di una certa soglia.
            -   es_delta : e' la soglia che l'errore di validazione deve superare entro un certo numero di epoche per capire se ci sono stati miglioramenti significativi nell'addestramento.

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
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe TrainingParams. Viene principalmente utilizzata per stampare in console i valori delle proprietà del layer con una formattazione più precisa.

            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'TrainingParams(\n\tbatch_size = {self.batch_size:.0f},\n\t epochs = {self.epochs:.0f},\n\t learning_rate = {self.learning_rate:.2f},\n\t rprop = {self.rprop},\n\t eta_minus = {self.eta_minus:.5f},\n\t eta_plus = {self.eta_plus:.5f},\n\t delta_min = {self.delta_min:.5f},\n\t delta_max = {self.delta_max:.5f},\n\t es_patience = {self.es_patience:.0f},\n\t es_delta = {self.es_delta:.5f}\n)'
    
    # end

# end class TrainingParams

# ########################################################################### #
# RIFERIMENTI

