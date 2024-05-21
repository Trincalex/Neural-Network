'''

    artificial_neuron.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di un neurone di una rete neurale shallow feed-forward fully-connected (aka. Multilayer Perceptron) tramite paradigma di programmazione a oggetti.
    In particolare, ogni oggetto della classe che implementa il neurone (Neuron) e' composto di un vettore per gli input ed un vettore per i pesi entrambi provenienti dalle connessioni precedenti, ed un bias.

'''

# ########################################################################### #
# LIBRERIE

import constants
import auxfunc
import numpy as np
import pprint

# ########################################################################### #
# VARIABILI GLOBALI

tot_neurons = 0

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE NEURONE

class Neuron:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def id(self) -> int:
        """È l'identificativo univoco del neurone."""
        return self._id
    # end

    @id.setter
    def id(self, value : int) -> None:
        self._id = value
    # end

    @property
    def neuron_size(self) -> int:
        """È la dimensione del vettore di input e dei pesi del neurone."""
        return self._neuron_size
    # end

    @neuron_size.setter
    def neuron_size(self, value : int) -> None:
        if (value <= 0):
            raise ValueError("La dimensione del neurone deve essere maggiore di 0.")
        
        self._neuron_size = value
    # end

    @property
    def inputs(self) -> np.ndarray:
        """È l'array di valori in input al neurone."""
        return self._inputs
    # end

    @inputs.setter
    def inputs(self, value : np.ndarray) -> None:
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore degli input deve essere di tipo 'numpy.ndarray'.")
        
        # print(self.id, len(value), self.neuron_size)
        if (not len(value) == self.neuron_size):
            raise ValueError("La dimensione del vettore degli input non corrisponde a quella del neurone.")
        
        self._inputs = value
    # end

    @property
    def weights(self) -> np.ndarray:
        """È l'array di pesi del neurone."""
        return self._weights
    # end

    @weights.setter
    def weights(self, value : np.ndarray) -> None:
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore dei pesi deve essere di tipo 'numpy.ndarray'.")
        
        # print(self.id, len(value), self.neuron_size)
        if (not len(value) == self.neuron_size):
            raise ValueError("La dimensione del vettore dei pesi non corrisponde a quella del neurone.")
        
        self._weights = value
    # end

    @property
    def bias(self) -> float:
        """È il bias del neurone che modifica la combinazione lineare di input e pesi."""
        return self._bias
    # end

    @bias.setter
    def bias(self, value : float) -> None:
        self._bias = value
    # end

    @property
    def out_val(self) -> float:
        """È lo scalare dato dalla somma del bias e della combinazione lineare di input e pesi."""
        return self._out_val
    # end

    @out_val.setter
    def out_val(self, value : float) -> None:
        self._out_val = value
    # end

    @property
    def act_fun(self) -> constants.ActivationFunctionType:
        """È la funzione di attivazione del neurone, con dominio e codominio a valori reali."""
        return self._act_fun
    # end

    @act_fun.setter
    def act_fun(self, fun : constants.ActivationFunctionType) -> None:
        self._act_fun = fun
    # end

    @property
    def act_val(self) -> float:
        """È il risultato dell'applicazione della funzione di attivazione."""
        return self._act_val
    # end

    @act_val.setter
    def act_val(self, value : float) -> None:
        self._act_val = value
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(self, ns : int) -> None:
        """
            È il costruttore della classe Neuron.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   ns : è la dimensione del neurone, cioe' la dimensione dei vettori di input e dei pesi

            Returns:
            -   None
        """

        global tot_neurons
        tot_neurons = tot_neurons + 1
        self.id = tot_neurons

        self.neuron_size = ns
        self.inputs = np.zeros(self.neuron_size)
        self.weights = np.zeros(self.neuron_size)

        self.bias = 0
        self.out_val = 0.0

        self.act_fun = auxfunc.identity
        self.act_val = 0.0

    # end

    # ####################################################################### #
    # METODI

    def output(self) -> float:
        """
            Calcola la somma tra il bias e la combinazione lineare di input e pesi.

            Returns:
            -   float : il valore di output del neurone
        """

        self.out_val = np.dot(self.inputs, self.weights) + self.bias
        return self.out_val
    
    # end

    def activate(self, train : bool = False) -> float | tuple[float, float]:
        """
            Applica la funzione di attivazione all'output del neurone.

            Parameters:
            -   train : serve a distinguere se l'applicazione del metodo è durante la fase di training o meno.

            Returns:
            -   se train=False, restituisce il valore di attivazione del neurone.
            -   se train=True, restituisce sia l'output intermedio prima dell'applicazione della funzione di attivazione sia il valore di attivazione del neurone.
        """

        out = self.output()
        self.act_val = self.act_fun(out)

        if train:
            return out, self.act_val
        
        return self.act_val
        # https://youtu.be/IHZwWFHWa-w
        
    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Neuron.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'Neuron(\n\tid = {self.id},\n\tsize = {self.neuron_size},\n\tinputs = {pprint.pformat(self.inputs)},\n\tweights = {pprint.pformat(self.weights)},\n\tbias = {self.bias},\n\tact_fun = {self.act_fun}\n)'
    
    # end

# end class Neuron