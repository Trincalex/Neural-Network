'''

    artificial_layer.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di un layer di una rete neurale shallow feed-forward fully-connected (aka. Multilayer Perceptron) tramite paradigma di programmazione a oggetti.
    In particolare, ogni oggetto della classe che implementa il layer (Layer) e' composto di uno o più neuroni (Neuron).

'''

# ########################################################################### #
# LIBRERIE

from artificial_neuron import Neuron

import constants
import auxfunc
import numpy as np
import pprint

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE LAYER

class Layer:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def units(self) -> list[Neuron]:
        """È la lista di neuroni del layer."""
        return self._units
    # end

    @units.setter
    def units(self, value : list[Neuron]) -> None:
        self._units = value
    # end

    @property
    def layer_size(self) -> int:
        """È la dimensione del layer, cioe' il numero di neuroni di cui e' composto."""
        return self._layer_size
    # end

    @layer_size.setter
    def layer_size(self, value : int) -> None:
        if (value <= 0):
            raise ValueError("La dimensione del layer deve essere maggiore di 0.")
        self._layer_size = value
    # end

    @property
    def weights(self) -> np.ndarray:
        """
            Restituisce i pesi del layer.
            
            Returns:
            -   una matrice contenente i pesi di tutti i neuroni del layer, dove la la prima dimensione indica il numero di neuroni del layer mentre la seconda dimensione indica il numero di pesi all'interno del singolo neurone.
        """
        
        return np.reshape([n.weights for n in self.units], (self.layer_size, self.units[0].neuron_size))
    
    # end

    @weights.setter
    def weights(self, value : np.ndarray) -> None:
        # print("Layer:", len(value), self.weights.size)
        if (len(value) <= 0 or len(value) > self.weights.size):
            raise ValueError("La dimensione del vettore dei pesi non e' compatibile.")

        start = 0
        end = 0
        for n in self.units:
            end += len(n.weights)
            n.weights = value[start:end]
            start = end

    # end

    @property
    def biases(self) -> np.ndarray:
        """
            Restituisce i bias del layer.
            
            Returns:
            -   un vettore colonna contenente i bias di tutti i neuroni del layer, dove la prima dimensione indica il numero di neuroni del layer mentre la seconda dimensione e' 1 perche' il bias e' uno scalare.
        """
        
        # Si utilizza -1 per recuperare la dimensione della lista originale.
        return np.reshape([n.bias for n in self.units], (-1,1))
    
    # end

    @biases.setter
    def biases(self, value : np.ndarray) -> None:
        # print("Layer:", len(value), len(self.biases))
        if (len(value) <= 0 or len(value) > len(self.biases)):
            raise ValueError("La dimensione del vettore dei bias non e' compatibile.")
        
        start = 0
        end = 0
        for n in self.units:
            end += 1
            n.bias = value[start]
            start = end

    # end

    @property
    def act_fun(self) -> constants.ActivationFunctionType:
        """È la funzione di attivazione di tutti i neuroni del layer, con dominio e codominio a valori reali."""
        return self._act_fun
    # end

    @act_fun.setter
    def act_fun(self, fun : constants.ActivationFunctionType) -> None:
        self._act_fun = fun
        for i in range(len(self.units)):
            self.units[i].act_fun = fun
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(self, ls : int, ns : int, af : constants.ActivationFunctionType) -> None:
        """
            È il costruttore della classe Layer.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   ls : e' la dimensione del layer, cioe' il numero di neuroni di cui e' composto.
            -   ns : e' la dimensione del neurone, cioe' la dimensione dei vettori di input e dei pesi.
            -   af : e' la funzione di attivazione da associare a tutti i neuroni del layer.

            Returns:
            -   None
        """

        # Richiama il setter della property "layer_size"
        self.layer_size = int(ls)

        self.units = []
        for i in range(ls):
            self.units.append(Neuron(ns))
        
        self.act_fun = af

    # end

    # ####################################################################### #
    # METODI

    def output(self) -> np.ndarray:
        """
            Calcola gli output dei neuroni del layer corrente.
            
            Returns:
            -   un numpy.ndarray contenente tutti gli output dei neuroni.
        """

        layer_outputs = [n.output() for n in self.units]
        return np.array(layer_outputs)

    # end

    def activate(self, train : bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
            Calcola i valori di attivazione dei neuroni del layer corrente.

            Parameters:
            -   train : serve a distinguere se l'applicazione del metodo è durante la fase di training o meno.

            Returns:
            -   se train=False, un numpy.ndarray contenente tutti i valori di attivazione dei neuroni.
            -   se train=True, restituisce sia un numpy.ndarray per gli output intermedi prima dell'applicazione della funzione di attivazione sia uno per i valori di attivazione.
            
        """

        layer_outputs = []
        layer_activations = []

        if train:
            for neuron in self.units:
                out, act = neuron.activate(train=train)
                layer_outputs.append(out)
                layer_activations.append(act)

            return np.array(layer_outputs), np.array(layer_activations)

        layer_activations = [neuron.activate() for neuron in self.units]
        return np.array(layer_activations)
    
    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Layer.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """

        return f'Layer(\n\tsize = {self.layer_size}\n)'
    
    # end

# end class Layer