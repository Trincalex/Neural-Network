'''

    artificial_neural_network.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di una rete neurale tramite paradigma di programmazione a oggetti. In particolare, la classe che implementa la rete neurale (NeuralNetwork) può essere composta di uno o più strati (Layer) che, a loro volta, possono essere composti di uno o più neuroni (Neuron).

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
    # end

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
    # end

    @out_val.setter
    def out_val(self, value):
        self._out_val = value
    # end

    # È la funzione di attivazione del neurone.
    @property
    def act_fun(self):
        return self._act_fun
    # end

    @act_fun.setter
    def act_fun(self, fun):
        self._act_fun = fun
    # end

    # È il risultato dell'applicazione della funzione di attivazione
    @property
    def act_val(self):
        return self._act_val
    # end

    @act_val.setter
    def act_val(self, value):
        self._act_val = value
    # end

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

    def output(self, train=False):
        """
            Calcola la somma tra il bias e la combinazione lineare di input e pesi e ne restituisce l'applicazione della funzione di attivazione.

            :param train: ...
            :return: l'output del neurone
        """

        self.out_val = np.dot(self.inputs, self.weights) + self.bias
        self.act_val = self.act_fun(self.out_val)

        if train:
            return self.out_val, self.act_val
        
        return self.act_val
    # end

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Neuron.
            
            :return: una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'Neuron(\n\tid = {self.id},\n\tsize = {self.neuron_size},\n\tinputs = {pprint.pformat(self.inputs)},\n\tweights = {pprint.pformat(self.weights)},\n\tbias = {self.bias},\n\tact_fun = {self.act_fun}\n)'
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
    # end

    @units.setter
    def units(self, value):
        self._units = value
    # end

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
    # COSTRUTTORE

    def __init__(self, ls, ns) -> None:
        self.layer_size = int(ls) # richiama il setter della property
        self.units = []

        for i in range(ls):
            self.units.append(Neuron(ns))
    # end

    # ####################################################################### #
    # METODI

    def output(self):
        """
            Calcola gli output dei neuroni del layer corrente.
            
            :return: un numpy.ndarray contenente tutti gli output dei neuroni.
        """
        
        layer_output = []

        for i in range(len(self.units)):
            n = self.units[i]
            layer_output.append(n.output())

        return np.array(layer_output)
    
    # end

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Layer.
            
            :return: una stringa contenente i dettagli dell'oggetto.
        """

        return f'Layer(\n\tsize = {self.layer_size}\n)'
    # end

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
    # end

    @depth.setter
    def depth(self, value):
        self._depth = value
    # end

    # È la dimensione del vettore di input della rete neurale.
    @property
    def input_size(self):
        return self._input_size
    # end

    @input_size.setter
    def input_size(self, value):
        if (value <= 0):
            raise ValueError("La dimensione dell'input della rete neurale deve essere maggiore di 0")
        
        self._input_size = int(value)
    # end

    # È l'array di valori in input alla rete neurale.
    @property
    def inputs(self):
        return self._inputs
    # end

    @inputs.setter
    def inputs(self, value):
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore degli input deve essere di tipo 'numpy.ndarray'.")
        
        if (not len(value) == self.input_size):
            raise ValueError("La dimensione del vettore degli input non e' compatibile.")
        
        self._inputs = value
    # end

    # È una lista di tutti i Layer nascosti della rete.
    @property
    def hidden_layers(self):
        return self._hidden_layers
    # end

    @hidden_layers.setter
    def hidden_layers(self, value):
        self._hidden_layers = value
    # end

    # È il Layer che raccoglie l'output complessivo della rete.
    @property
    def output_layer(self):
        return self._output_layer
    # end

    @output_layer.setter
    def output_layer(self, value):
        self._output_layer = value
    # end

    # ...
    @property
    def network_error(self):
        return self._network_error
    # end

    @network_error.setter
    def network_error(self, value):
        self._network_error = value
    # end

    # ...
    @property
    def average_error(self):
        return self._average_error
    # end

    @average_error.setter
    def average_error(self, value):
        self._average_error = value
    # end

    # ...
    @property
    def network_accuracy(self):
        return self._network_accuracy
    # end

    @network_accuracy.setter
    def network_accuracy(self, value):
        self._network_accuracy = value
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(self, i_size, hidden_sizes, output_size) -> None:
        
        # Inizializzazione dell'input
        self.input_size = i_size
        self.inputs = np.zeros(self.input_size)

        # Calcolo del numero di layers
        h_sizes = []
        h_sizes.insert(0, self.input_size)
        if np.isscalar(hidden_sizes):
            h_sizes.append(hidden_sizes)
        else:
            for hs in hidden_sizes:
                h_sizes.append(hs)

        # Inizializzazione degli hidden layers
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

        # Inizializzazione dell'output layer
        self.output_layer = Layer(output_size, h_sizes[-1])
        for j in range(len(self.output_layer.units)):
            n = self.output_layer.units[j]
            n.weights = np.array(constants.STANDARD_DEVIATION * np.random.normal(size=h_sizes[-1]))
            n.bias = constants.STANDARD_DEVIATION * np.random.normal()
            n.act_fun = auxfunc.sigmoid

        '''
            La profondita' della rete e' data dal numero di layer totali.
            Dato che 'h_sizes' tiene conto della dimensione dell'input e di tutte le dimensioni degli hidden layers, per contare la profondita' e' necessario togliere (input) e aggiungere (output) il valore 1.
        '''
        self.depth = len(h_sizes) - 1 + 1
        
        self.network_error = 0.0
        self.network_accuracy = 0.0
        self.average_error = 0.0

    # end
    
    # ####################################################################### #
    # METODI

    def load_input(self, x : list[float]) -> None:
        """
            Copia il vettore 'x' nel vettore 'inputs' della rete neurale.

            :param x: il vettore di dati in input da caricare nella rete neurale
            :return: None
        """

        if (len(x) > self.input_size):
            raise ValueError("La dimensione del vettore degli input non e' compatibile.")
        
        self.inputs = np.array(x)
            
    # end

    def forward_propagation(self,
                            x : list[float],
                            train=False):
        """
            Calcola l'output complessivo della rete neurale, propagando i dati attraverso le connessioni di input dell'input layer, attraverso i calcoli intermedi degli hidden layers e, infine, attraverso l'ultimo strato dell'output layer.
            
            :return: se train=False, un numpy.ndarray contenente l'output complessivo della rete.
            :return: se train=True, una lista contenente gli output di tutti gli strati della rete.
        """

        out = []

        # Carica input nella rete neurale
        self.load_input(x)

        # Passa input al primo hidden layer
        # out.append(self.input_layer.output())
        out.append(self.inputs)
        pprint.pprint(out[-1])

        for i in range(len(self.hidden_layers)):
            hl = self.hidden_layers[i]

            # Aggiorna input dell'i-esimo hidden layer
            for j in range(len(hl.units)):
                n = hl.units[j]
                n.inputs = out[-1]
            
            # Calcola output dell'i-esimo hidden layer
            out.append(hl.output())

        pprint.pprint(out[-1])

        # Aggiorna input dell'output layer
        for j in range(len(self.output_layer.units)):
            n = self.output_layer.units[j]
            n.inputs = out[-1]

        # Calcola output dell'output layer
        out.append(self.output_layer.output())
        pprint.pprint(out[-1])

        if train:
            return out

        return out[-1]
    
    # end

    def back_propagation(self,
                         x : list[float],
                         y : list[float],
                         learning_rate):
        """
            E' un algoritmo iterativo utilizzato per l'addestramento delle reti neurali. Serve a minimizzare la funzione di costo () determinando quali pesi e quali bias devono essere modificati. La sua implementazione si basa sulla 'regola della catena', utile a navigare attraverso i molteplici strati della rete neurale.
            
            :param x: ...
            :param y: ...
            :param learning_rate: ...
            :return: 
        """

        all_outputs = self.forward_propagation(x, train=True)
        predicted_output = all_outputs[-1]
        output_error = y - predicted_output
        output_delta = output_error - auxfunc.sigmoid(predicted_output, der=True)

        print(output_error)
        # TODO: da completare
        
    # end

    def resilient_back_propagation(self):
        """
            ...
            
            :param ...: ...
            :return: 
        """
        
        pass
    # end

    def train(self):
        """
            ...
            
            :param ...: ...
            :return: 
        """

        pass
    # end

    def predict(self):
        """
            ...
            
            :param ...: ...
            :return: 
        """
        
        pass
    # end

    def compute_accuracy(self):
        """
            ...
            
            :param ...: ...
            :return: 
        """
        
        pass
    # end

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe NeuralNetwork.
            
            :return: una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'NeuralNetwork(\n\tdepth = {self.depth},\n\tinput_size = {repr(self.input_size)},\n\tinputs = {pprint.pformat(self.inputs)},\n\thidden_layers = {pprint.pformat(self.hidden_layers)},\n\toutput_layer = {self.output_layer},\n\tnetwork_error = {self.network_error},\n\tnetwork_accuracy = {self.network_accuracy},\n\taverage_error = {self.average_error}\n)'
    # end

    # end class NeuralNetwork

# ########################################################################### #
# RIFERIMENTI

# https://medium.com/@geertvandamme/building-an-object-oriented-neural-network-ee3f4af085b6
# https://www.toptal.com/python/python-class-attributes-an-overly-thorough-guide
# https://compphysics.github.io/MachineLearningMSU/doc/pub/NeuralNet/html/._NeuralNet-bs048.html
# https://stackoverflow.com/questions/40185437/no-module-named-numpy-visual-studio-code
# https://stackoverflow.com/questions/2627002/whats-the-pythonic-way-to-use-getters-and-setters
# https://www.digitalocean.com/community/tutorials/python-str-repr-functions
# https://testdriven.io/blog/documenting-python/
# https://stackoverflow.com/questions/32514502/neural-networks-what-does-the-input-layer-consist-of