'''

    artificial_neural_network.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    ...

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

    # È l'identificativo univoco del neurone.
    @property
    def id(self):
        return self._id
    # end

    @id.setter
    def id(self, value):
        self._id = value

    # È la dimensione del vettore di input e dei pesi del neurone.
    @property
    def neuron_size(self):
        return self._neuron_size
    # end

    @neuron_size.setter
    def neuron_size(self, value):
        if (value <= 0):
            raise ValueError("La dimensione del neurone deve essere maggiore di 0")
        
        self._neuron_size = int(value)
    # end

    # È l'array di valori in input al neurone.
    @property
    def inputs(self):
        return self._inputs
    # end

    @inputs.setter
    def inputs(self, value):
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore degli input deve essere di tipo 'numpy.ndarray'.")
        
        # print(self.id, len(value), self.neuron_size)
        if (not len(value) == self.neuron_size):
            raise ValueError("La dimensione del vettore degli input non corrisponde a quella del neurone.")
        
        self._inputs = value
    # end

    # È l'array di pesi del neurone.
    @property
    def weights(self):
        return self._weights
    # end

    @weights.setter
    def weights(self, value):
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore dei pesi deve essere di tipo 'numpy.ndarray'.")
        
        # print(self.id, len(value), self.neuron_size)
        if (not len(value) == self.neuron_size):
            raise ValueError("La dimensione del vettore dei pesi non corrisponde a quella del neurone.")
        
        self._weights = value
    # end

    # È il bias del neurone che modifica la combinazione lineare di input e pesi.
    @property
    def bias(self):
        return self._bias
    # end

    @bias.setter
    def bias(self, value):
        self._bias = float(value)
    # end

    # È lo scalare dato dalla somma del bias e della combinazione lineare di input e pesi.
    @property
    def out_val(self):
        return self._out_val

    @out_val.setter
    def out_val(self, value):
        self._out_val = value

    # È la funzione di attivazione del neurone.
    @property
    def act_fun(self):
        return self._act_fun

    @act_fun.setter
    def act_fun(self, fun):
        self._act_fun = fun

    # È il risultato dell'applicazione della funzione di attivazione
    @property
    def act_val(self):
        return self._act_val

    @act_val.setter
    def act_val(self, value):
        self._act_val = value

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(self, ns) -> None:
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

    '''
        Calcola la somma tra il bias e la combinazione lineare di input e pesi e ne restituisce l'applicazione della funzione di attivazione.
    '''
    def output(self, train=False):
        self.out_val = np.dot(self.inputs, self.weights) + self.bias
        if not self.out_val > 0:
            self.out_val = 0
        self.act_val = self.act_fun(self.out_val)

        if train:
            return self.out_val, self.act_val
        
        return self.act_val

    # end

    '''
        Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Neuron.
    '''
    def __repr__(self) -> str:
        return f'Neuron(\n\tid={self.id},\n\tsize={self.neuron_size},\n\tinputs={pprint.pformat(self.inputs)},\n\tweights={pprint.pformat(self.weights)},\n\tbias={self.bias},\n\tact_fun={self.act_fun}\n)'
    # end

    # end class Neuron

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE LAYER

class Layer:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    # È la lista di neuroni del layer
    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, value):
        self._units = value

    # È la dimensione del layer
    @property
    def layer_size(self):
        return self._layer_size
    # end

    @layer_size.setter
    def layer_size(self, value):
        if (value > 0):
            self._layer_size = int(value)
        else:
            raise ValueError("La dimensione del layer deve essere maggiore di 0")
    # end

    # ####################################################################### #
    # METODI

    def __init__(self, ls, ns) -> None:
        self.layer_size = int(ls) # richiama il setter della property
        self.units = []

        for i in range(ls):
            self.units.append(Neuron(ns))
    # end

    '''
        Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Layer.
    '''
    def __repr__(self) -> str:
        pass

    # end class Layer

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE NEURAL NETWORK

class NeuralNetwork:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    # È la profondità della rete, cioe' il numero totale di Layer.
    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value

    # È il Layer che raccoglie tutte le connessioni in input nella rete.
    @property
    def input_layer(self):
        return self._input_layer

    @input_layer.setter
    def input_layer(self, value):
        self._input_layer = value

    # È una lista di tutti i Layer nascosti della rete.
    @property
    def hidden_layers(self):
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers(self, value):
        self._hidden_layers = value

    # È il Layer che raccoglie l'output complessivo della rete.
    @property
    def output_layer(self):
        return self._output_layer

    @output_layer.setter
    def output_layer(self, value):
        self._output_layer = value

    # ...
    @property
    def network_error(self):
        return self._network_error

    @network_error.setter
    def network_error(self, value):
        self._network_error = value

    # ...
    @property
    def average_error(self):
        return self._average_error

    @average_error.setter
    def average_error(self, value):
        self._average_error = value

    # ...
    @property
    def network_accuracy(self):
        return self._network_accuracy

    @network_accuracy.setter
    def network_accuracy(self, value):
        self._network_accuracy = value

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(self, input_size, hidden_sizes, output_size) -> None:
        
        # inizializzazione dell'input layer
        self.input_layer = Layer(input_size, constants.DIMENSIONE_NEURONE_INPUT)
        for i in range(len(self.input_layer.units)):
            n = self.input_layer.units[i]
            n.weights = np.ones(constants.DIMENSIONE_NEURONE_INPUT)
            n.bias = 0.0
            n.act_fun = auxfunc.identity

        # calcolo del numero di layers
        h_sizes = []
        h_sizes.insert(0, input_size)
        if np.isscalar(hidden_sizes):
            h_sizes.append(hidden_sizes)
        else:
            for hs in hidden_sizes:
                h_sizes.append(hs)

        # inizializzazione degli hidden layers
        self.hidden_layers = []
        for i in range(1, len(h_sizes)):
            # print(f'Hidden layer n.{i}')
            prev_size = h_sizes[i-1]
            actual_size = h_sizes[i]
            hl = Layer(actual_size, prev_size)

            for j in range(len(hl.units)):
                # print(f'Neuron n.{j}')
                n = hl.units[j]
                n.weights = np.array(constants.STANDARD_DEVIATION * np.random.normal(size=prev_size))
                n.bias = constants.STANDARD_DEVIATION * np.random.normal()
                n.act_fun = auxfunc.sigmoid

            self.hidden_layers.append(hl)

        # inizializzazione dell'output layer
        self.output_layer = Layer(output_size, h_sizes[-1])
        for i in range(len(self.output_layer.units)):
            n = self.output_layer.units[i]
            n.weights = np.array(constants.STANDARD_DEVIATION * np.random.normal(size=h_sizes[-1]))
            n.bias = constants.STANDARD_DEVIATION * np.random.normal()
            n.act_fun = auxfunc.sigmoid

        '''
            La profondita' della rete e' data dal numero di layer totali.
            Dato che 'h_sizes' tiene gia' conto dell'input layer e di tutti gli hidden layers, per contare l'output layer è necessario aggiungere solo '1'.
        '''
        self.depth = len(h_sizes) + 1

    # end
    
    # ####################################################################### #
    # METODI

    def load_input(self, x):
        for i in range(len(self.input_layer.units)):
            n = self.input_layer.units[i]
            n.inputs[0] = x[i]
    # end

    def forward_propagation(self):
        # calcola output neuroni input layer

        # calcola output neuroni 1° hidden layer
        # ...
        # calcola output neuroni n-esimo hidden layer

        # calcola output neuroni output layer

        pass
    
    # end

    def resilient_back_propagation(self):
        pass
    # end

    def train(self):
        pass
    # end

    def predict(self):
        pass
    # end

    def compute_accuracy(self):
        pass
    # end

    '''
        Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe NeuralNetwork.
    '''
    def __repr__(self) -> str:
        pass

    # end class NeuralNetwork

# ########################################################################### #
# RIFERIMENTI

# https://medium.com/@geertvandamme/building-an-object-oriented-neural-network-ee3f4af085b6
# https://www.toptal.com/python/python-class-attributes-an-overly-thorough-guide
# https://compphysics.github.io/MachineLearningMSU/doc/pub/NeuralNet/html/._NeuralNet-bs048.html
# https://stackoverflow.com/questions/40185437/no-module-named-numpy-visual-studio-code
# https://stackoverflow.com/questions/2627002/whats-the-pythonic-way-to-use-getters-and-setters
# https://www.digitalocean.com/community/tutorials/python-str-repr-functions