"""

    artificial_layer.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di un layer di una rete neurale artificiale feed-forward fully-connected (e.g. Multilayer Perceptron) tramite paradigma di programmazione a oggetti.
    La classe Layer e' il cuore della rete neurale. Rappresenta un singolo livello di neuroni. Gestisce in modo indipendente rispetto al resto della rete neurale i valori dei pesi / bias e le operazioni di attivazione per il livello specifico, facilitando il monitoraggio della propagazione dei dati in avanti (forward-propagation) e all'indietro (back-propagation) attraverso la rete neurale.

"""

# ########################################################################### #
# LIBRERIE

import constants
import numpy as np
import pprint

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE LAYER

class Layer:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def layer_size(self) -> int:
        """E' la dimensione del layer, cioe' il numero di neuroni di cui e' composto."""
        return self._layer_size
    # end

    @property
    def neuron_size(self) -> int:
        """E' la dimensione di un neurone del layer, cioe' il numero di connessioni provenienti dal livello precedente."""
        return self._neuron_size
    # end

    # @layer_size.setter
    # def layer_size(self, value : int) -> None:
    #     if (value <= 0):
    #         raise ValueError("La dimensione del layer deve essere maggiore di 0.")
    #     self._layer_size = value
    # # end

    @property
    def inputs(self) -> np.ndarray:
        """ E' la matrice di valori in input al layer. Ha un numero di colonne pari al numero di neuroni del layer precedente ed un numero di righe pari al numero di esempi in input alla rete neurale. """

        return self._inputs
    # end

    @inputs.setter
    def inputs(self, value : np.ndarray) -> None:
        if (not isinstance(value, np.ndarray)):
            raise constants.LayerError("Il matrice degli input deve essere di tipo 'numpy.ndarray'.")
        
        # print(value.shape, self.neuron_size)
        if len(value.shape) == 2:
            if (not value.shape[1] == self.neuron_size):
                raise constants.LayerError("La dimensione dei vettori delle caratteristiche degli input non e' compatibile con questo layer.")
        elif len(value.shape) == 1:
            if (not value.size == self.neuron_size):
                raise constants.LayerError("La dimensione del vettore delle caratteristiche dell'input non e' compatibile con questo layer.")
        else:
            raise constants.LayerError("La matrice degli input non e' compatibile con questo layer.")
        
        self._inputs = value
    # end

    @property
    def weights(self) -> np.ndarray:
        """
            E' la matrice dei pesi del layer.
            
            Returns:
            -   una matrice contenente i pesi di tutti i neuroni del layer, dove la prima dimensione indica il numero di neuroni del layer mentre la seconda dimensione indica il numero di pesi all'interno del singolo neurone.
        """
        
        return self._weights
    
    # end

    @weights.setter
    def weights(self, value : np.ndarray) -> None:

        if (not isinstance(value, np.ndarray)):
            raise constants.LayerError("La matrice dei pesi deve essere di tipo 'numpy.ndarray'.")
        
        # print("Layer:", value.shape, self.weights.shape)
        if (not value.shape == self._weights.shape):
            raise constants.LayerError("La matrice dei pesi non e' compatibile con questo layer.")
        
        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- LAYER (weights) ---\n")
                print(value.shape)
                pprint.pprint(value)
                print("\n-----\n\n")

        self._weights = value

    # end

    @property
    def biases(self) -> np.ndarray:
        """
            E' il vettore colonna dei bias del layer.
            
            Returns:
            -   un vettore colonna contenente i bias di tutti i neuroni del layer, dove la prima dimensione indica il numero di neuroni del layer mentre la seconda dimensione e' 1 perche' il bias e' uno scalare.
        """

        return self._biases
    
    # end

    @biases.setter
    def biases(self, value : np.ndarray) -> None:
        
        if (not isinstance(value, np.ndarray)):
            raise constants.LayerError("Il vettore dei bias deve essere di tipo 'numpy.ndarray'.")
        
        # print("Layer:", value.shape, self.biases.shape)
        if (not value.shape == self._biases.shape):
            raise constants.LayerError("Il vettore dei bias non e' compatibile con questo layer.")
        
        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- LAYER (biases) ---\n")
                print(value.shape)
                pprint.pprint(value)
                print("\n-----\n\n")

        self._biases = value

    # end

    @property
    def act_fun(self) -> constants.ActivationFunctionType:
        """E' la funzione di attivazione di tutti i neuroni del layer, con dominio e codominio a valori reali."""
        return self._act_fun
    # end

    @act_fun.setter
    def act_fun(self, fun : constants.ActivationFunctionType) -> None:
        self._act_fun = fun
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            ls : int,
            ns : int,
            af : constants.ActivationFunctionType,
            random_init : bool = True
    ) -> None:
        
        """
            E' il costruttore della classe Layer.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   ls : e' la dimensione del layer, cioe' il numero di neuroni di cui e' composto.
            -   ns : e' la dimensione del neurone, cioe' la dimensione dei vettori di input e dei pesi.
            -   af : e' la funzione di attivazione da associare a tutti i neuroni del layer.

            Returns:
            -   None
        """

        # Inizializzazione delle dimensioni del layer
        if (ls <= 0):
            raise constants.LayerError("La dimensione del layer deve essere maggiore di 0.")
        self._layer_size = int(ls)
        
        if (ns <= 0):
            raise constants.LayerError("La dimensione del neurone deve essere maggiore di 0.")
        self._neuron_size = int(ns)

        # Inizializzazione dell'input del layer
        self.inputs = np.zeros((1,ns))

        # Inizializzazione dei pesi / bias del layer
        if not random_init:
            rng = np.random.default_rng(constants.DEFAULT_RANDOM_SEED)

        self._weights = np.zeros((ls,ns))
        if not random_init:
            self.weights = rng.normal(
                loc=constants.DEFAULT_DISTRIBUTION_MEAN,
                scale=constants.DEFAULT_STANDARD_DEVIATION,
                size=(ls,ns)
            )
        else:
            self.weights = np.random.normal(
                loc=constants.DEFAULT_DISTRIBUTION_MEAN, 
                scale=constants.DEFAULT_STANDARD_DEVIATION, 
                size=(ls,ns)
            )

        self._biases = np.zeros((ls,1))
        if not random_init:
            self.biases = rng.normal(
                loc=constants.DEFAULT_DISTRIBUTION_MEAN, 
                scale=constants.DEFAULT_STANDARD_DEVIATION, 
                size=(ls,1)
            )
        else:
            self.biases = np.random.normal(
                loc=constants.DEFAULT_DISTRIBUTION_MEAN, 
                scale=constants.DEFAULT_STANDARD_DEVIATION, 
                size=(ls,1)
            )
        
        self.act_fun = af

    # end

    # ####################################################################### #
    # METODI

    def output(self) -> np.ndarray:
        """
            Calcola gli output (o input pesati) dei neuroni del layer corrente come una combinazione lineare dei valori in input al neurone e i pesi lungo le connessioni aggiungendo, infine, il bias.
            
            Returns:
            -   numpy.ndarray : un array contenente tutti gli input pesati dei neuroni del layer corrente.
        """

        layer_outputs = np.dot(self.inputs, self.weights.T) + self.biases.T

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- LAYER PROPAGATION (outputs) ---\n")
                # print(self.inputs.shape)
                # print(self.weights.T.shape)
                # print(self.biases.T.shape)
                print(layer_outputs.shape)
                pprint.pprint(layer_outputs)
                print("\n-----\n\n")
        
        return layer_outputs

    # end

    def activate(self, train : bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
            Calcola i valori di attivazione dei neuroni del layer corrente.

            Parameters:
            -   train : serve a distinguere se l'applicazione del metodo e' durante la fase di training o meno.

            Returns:
            -   se train=False, un numpy.ndarray contenente tutti i valori di attivazione dei neuroni.
            -   se train=True, restituisce sia un numpy.ndarray per gli output intermedi prima dell'applicazione della funzione di attivazione sia un altro per i valori di attivazione.
        """

        layer_outputs = self.output()
        layer_activations = self.act_fun(layer_outputs)

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- LAYER PROPAGATION (activations) ---\n")
                print(layer_activations.shape)
                pprint.pprint(layer_activations)
                print("\n-----\n\n")
                
        if train:
            return layer_outputs, layer_activations

        return layer_activations

    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Layer. Viene principalmente utilizzata per stampare in console i valori delle proprietà del layer con una formattazione più precisa.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """

        return f'Layer(\n\tsize = {self.layer_size},\n\tact_fun = {self.act_fun},\n\tinputs_size = {self.inputs.shape}\n\tweights_shape = {self.weights.shape},\n\tbiases_shape = {self.biases.shape}\n)'
    
    # end

# end class Layer

# ########################################################################### #
# RIFERIMENTI

# https://saturncloud.io/blog/applying-a-function-along-a-numpy-array-a-comprehensive-guide-for-data-scientists/