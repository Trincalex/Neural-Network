"""

    training_report.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione della classe TrainingReport, il cui scopo e' quello di memorizzare tutte le misure di valutazione riguardanti l'addestramento di una rete neurale feed-forward fully-connected.

"""

# ########################################################################### #
# LIBRERIE

import constants
import numpy as np

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE TrainingReport

class TrainingReport:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def num_epochs(self) -> int:
        """ Il numero di epoche impiegate dalla fase di addestramento. Questo valore e' utile per tracciare l'evoluzione del training, soprattutto nel plotting dei grafici relativi alla fase di addestramento. """
        return self._num_epochs
    # end

    @property
    def elapsed_time(self) -> float:
        """ Il tempo totale impiegato per completare il numero di epoche indicato dall'attributo di classe 'num_epochs', utile per valutare l'efficienza del processo di addestramento. """
        return self._elapsed_time
    # end

    @property
    def training_examples(self) -> int:
        """ Il numero di esempi del training set utilizzati per la fase di addestramento. Questo valore risulta utile per confrontare le prestazioni ottenute da modelli allenati su una quantita' di esempi di training diversa. """
        return self._training_examples
    # end

    @property
    def training_error(self) -> float:
        """ L'errore di addestramento ottenuto al termine della fase di addestramento dopo un numero di epoche indicate dall'attributo di classe 'num_epochs'. Questo valore viene rappresentato graficamente nei plot delle curve di apprendimento. """
        return self._training_error
    # end

    @property
    def training_accuracy(self) -> float:
        """ L'accuracy di addestramento ottenuta al termine della fase di addestramento dopo un numero di epoche indicate dall'attributo di classe 'num_epochs'. Questo valore viene rappresentato graficamente nei plot delle curve di apprendimento. """
        return self._training_accuracy
    # end

    @property
    def validation_examples(self) -> int:
        """ Il numero di esempi del validation set utilizzati per la validazione del modello. Proprio come per l'attributo 'training_examples', anche questo valore e' utile per confrontare le prestazioni ottenute da modelli diversi. """
        return self._validation_examples
    # end

    @property
    def validation_error(self) -> float:
        """ L'errore di validazione ottenuto al termine della validazione del modello dopo un numero di epoche indicate dall'attributo di classe 'num_epochs'. Questo valore viene rappresentato graficamente nei plot delle curve di apprendimento. """
        return self._validation_error
    # end

    @property
    def validation_accuracy(self) -> float:
        """ L'accuracy di validazione ottenuta al termine della validazione del modello dopo un numero di epoche indicate dall'attributo di classe 'num_epochs'. Questo valore viene rappresentato graficamente nei plot delle curve di apprendimento. """
        return self._validation_accuracy
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            epochs : int        = 0,
            time : float        = 0.0,
            t_examples : int    = 0,
            v_examples : int    = 0,
            t_error : float     = np.inf,
            v_error : float     = np.inf,
            t_accuracy : float  = 0.0,
            v_accuracy : float  = 0.0
    ):
        """
            E' il costruttore della classe TrainingReport.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   epochs : e' il numero di epoche impiegate dalla fase di addestramento. 
            -   time : e' il tempo impiegato dalla fase di addestramento.
            -   t_examples : e' il numero di esempi del training set utilizzati per la fase di addestramento.
            -   v_examples : e' il numero di esempi del validation set utilizzati per la fase di validazione.
            -   t_error : e' l'errore di addestramento ottenuto al termine della fase di addestramento.
            -   v_error : e' l'errore di validazione ottenuto al termine della fase di validazione.
            -   t_accuracy : e' l'accuracy di addestramento ottenuta al termine della fase di addestramento.
            -   v_accuracy : e' l'accuracy di validazione ottenuta al termine della fase di validazione.

            Returns:
            -   None.
        """

        # Inizializzazione del numero di epoche.
        self._num_epochs = epochs

        # Inizializzazione del tempo di addestramento.
        self._elapsed_time = time

        # Inizializzazione del numero di esempi di addestramento e validazione.
        self._training_examples = t_examples
        self._validation_examples = v_examples

        # Inizializzazione delle metriche di errore.
        self._training_error = t_error
        self._validation_error = v_error

        # Inizializzazione delle metriche di accuracy.
        self._training_accuracy = t_accuracy
        self._validation_accuracy = v_accuracy
    
    # end

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
            -   float : il rapporto tra predizioni corrette e totale delle etichette.
        """

        # print(predictions.shape, targets.shape)
        if not predictions.shape == targets.shape:
            raise constants.TrainError("Il numero di predizioni e di etichette non sono compatibili.")

        matches = [int(np.argmax(x) == np.argmax(y)) for x, y in zip(predictions, targets)]
        return np.sum(matches) / len(targets)

    # end

    def update(self, value) -> None:
        """
            Aggiorna i valori del report corrente con i valori provenienti da un altro oggetto della classe TrainingReport.

            Parameters:
            -   value : e' un TrainingReport contenente i nuovi valori da copiare nel report corrente.

            Returns:
            -   None.
        
        """

        if (not isinstance(value, type(self))):
            raise TypeError("Impossibile aggiornare il report.")

        self._num_epochs            = value.num_epochs
        self._elapsed_time          = value.elapsed_time
        self._training_examples     = value.training_examples
        self._validation_examples   = value.validation_examples
        self._training_error        = value.training_error
        self._validation_error      = value.validation_error
        self._training_accuracy     = value.training_accuracy
        self._validation_accuracy   = value.validation_accuracy

    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe TrainingReport. Viene principalmente utilizzata per stampare in console i valori delle proprietà del layer con una formattazione più precisa.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'TrainingReport(\n\tnum_epochs = {self.num_epochs},\n\telapsed_time = {self.elapsed_time:.3f} secondi,\n\ttraining_examples = {self.training_examples},\n\ttraining_error = {self.training_error:.5f},\n\ttraining_accuracy = {self.training_accuracy:.2%},\n\tvalidation_examples = {self.validation_examples},\n\tvalidation_error = {self.validation_error:.5f},\n\tvalidation_accuracy = {self.validation_accuracy:.2%}\n)'
    
    # end

# end TrainingReport

# ########################################################################### #
# RIFERIMENTI

