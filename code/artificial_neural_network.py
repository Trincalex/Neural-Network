'''

    artificial_neural_network.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di una rete neurale shallow feed-forward (aka. Multilayer Perceptron) tramite paradigma di programmazione a oggetti. In particolare, la classe che implementa la rete neurale (NeuralNetwork) può essere composta di uno o più strati (Layer) che, a loro volta, possono essere composti di uno o più neuroni (Neuron).

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
            raise ValueError("La dimensione del neurone deve essere maggiore di 0")
        
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
            -   ns: è la dimensione del neurone, cioe' la dimensione dei vettori di input e dei pesi

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

    def output(self, train : bool = False) -> float:
        """
            Calcola la somma tra il bias e la combinazione lineare di input e pesi e ne restituisce l'applicazione della funzione di attivazione.

            Parameters:
            -   train: serve a distinguere se l'applicazione del metodo è durante la fase di training o meno.

            Returns:
            -   se train=False, restituisce l'output del neurone.
            -   se train=True, restituisce sia l'output del neurone che anche il valore intermedio prima dell'applicazione della funzione di attivazione.
        """

        self.out_val = np.dot(self.inputs, self.weights) + self.bias
        self.act_val = self.act_fun(self.out_val)

        if train:
            return self.out_val, self.act_val
        
        return self.act_val
        # https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=2
        
    # end

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe Neuron.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'Neuron(\n\tid = {self.id},\n\tsize = {self.neuron_size},\n\tinputs = {pprint.pformat(self.inputs)},\n\tweights = {pprint.pformat(self.weights)},\n\tbias = {self.bias},\n\tact_fun = {self.act_fun}\n)'
    
    # end

# end class Neuron

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
            raise ValueError("La dimensione del layer deve essere maggiore di 0")
        self._layer_size = value
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(self, ls : int, ns : int) -> None:
        """
            È il costruttore della classe Layer.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   ls: è la dimensione del layer, cioe' il numero di neuroni di cui e' composto.
            -   ns: è la dimensione del neurone, cioe' la dimensione dei vettori di input e dei pesi.

            Returns:
            -   None
        """

        # Richiama il setter della property "layer_size"
        self.layer_size = int(ls)

        self.units = []
        for i in range(ls):
            self.units.append(Neuron(ns))

    # end

    # ####################################################################### #
    # METODI

    def output(self) -> np.ndarray:
        """
            Calcola gli output dei neuroni del layer corrente.
            
            Returns:
            -   un numpy.ndarray contenente tutti gli output dei neuroni.
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
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """

        return f'Layer(\n\tsize = {self.layer_size}\n)'
    
    # end

# end class Layer

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE NEURAL NETWORK

class NeuralNetwork:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def depth(self) -> int:
        """È la profondità della rete, cioe' il numero totale di Layer."""
        return self._depth
    # end

    @depth.setter
    def depth(self, value : int) -> None:
        self._depth = value
    # end

    @property
    def input_size(self) -> int:
        """È la dimensione del vettore di input della rete neurale."""
        return self._input_size
    # end

    @input_size.setter
    def input_size(self, value : int) -> None:
        if (value <= 0):
            raise ValueError("La dimensione dell'input della rete neurale deve essere maggiore di 0")
        
        self._input_size = value
    # end

    @property
    def inputs(self) -> np.ndarray:
        """È l'array di valori in input alla rete neurale."""
        return self._inputs
    # end

    @inputs.setter
    def inputs(self, value : np.ndarray) -> None:
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore degli input deve essere di tipo 'numpy.ndarray'.")
        
        if (not len(value) == self.input_size):
            raise ValueError("La dimensione del vettore degli input non e' compatibile.")
        
        self._inputs = value
    # end

    @property
    def hidden_layers(self) -> list[Layer]:
        """È una lista di tutti i Layer nascosti della rete."""
        return self._hidden_layers
    # end

    @hidden_layers.setter
    def hidden_layers(self, value : list[Layer]) -> None:
        self._hidden_layers = value
    # end

    @property
    def output_layer(self) -> Layer:
        """È il Layer che raccoglie l'output complessivo della rete."""
        return self._output_layer
    # end

    @output_layer.setter
    def output_layer(self, value : Layer) -> None:
        self._output_layer = value
    # end

    @property
    def err_fun(self) -> constants.ErrorFunctionType:
        """È la funzione di errore utilizzata per verificare la qualità della rete neurale."""
        return self._err_fun
    # end

    @err_fun.setter
    def err_fun(self, value : constants.ErrorFunctionType) -> None:
        self._err_fun = value
    # end

    @property
    def network_error(self):
        """..."""
        return self._network_error
    # end

    @network_error.setter
    def network_error(self, value) -> None:
        self._network_error = value
    # end

    @property
    def average_error(self):
        """..."""
        return self._average_error
    # end

    @average_error.setter
    def average_error(self, value) -> None:
        self._average_error = value
    # end

    @property
    def network_accuracy(self):
        """..."""
        return self._network_accuracy
    # end

    @network_accuracy.setter
    def network_accuracy(self, value) -> None:
        self._network_accuracy = value
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            i_size : int,
            hidden_sizes : list[int],
            output_size : int,
            hidden_act_funs : list[constants.ActivationFunctionType] = [auxfunc.sigmoid],
            output_act_fun : constants.ActivationFunctionType = auxfunc.sigmoid,
            e_fun : constants.ErrorFunctionType = auxfunc.cross_entropy
    ) -> None:
        
        """
            È il costruttore della classe NeuralNetwork.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   i_size: è la dimensione del vettore in input alla rete neurale
            -   hidden_sizes: può essere un numero o una lista contenente la dimensione di uno o più-  hidden layer della rete neurale.
            -   output_size: è la dimensione dell'output layer della rete neurale.
            -   hidden_act_funs: può essere una funzione o una lista contenente le funzioni di attivazione di uno o più hidden layer della rete neurale.
            -   output_act_fun: è la funzione di attivazione dei neuroni dell'output layer.
            -   e_fun: è la funzione di errore utilizzata per verificare la qualità della rete neurale.

            Returns:
            -   None
        """
        
        # Inizializzazione dell'input
        self.input_size = i_size
        self.inputs = np.zeros(self.input_size)

        # Inserimento delle dimensioni dell'input e degli hidden layer in una lista
        l_sizes = []
        l_sizes.insert(0, self.input_size)
        if np.isscalar(hidden_sizes):
            l_sizes.append(hidden_sizes)
        else:
            for hs in hidden_sizes:
                l_sizes.append(hs)
        
        # Inserimento delle funzioni di attivazioni degli hidden e dell'output layer in una lista
        l_act_funs = []
        if not isinstance(hidden_act_funs, list):
            l_act_funs.append(hidden_act_funs)
        else:
            for hf in hidden_act_funs:
                l_act_funs.append(hf)
        l_act_funs.append(output_act_fun)

        # Controllo sul numero di hidden layer e funzioni di attivazioni inserite
        if (len(l_sizes) != len(l_act_funs)):
            raise constants.HiddenLayerError("Il numero di funzioni di attivazione deve essere uguale al numero di layer!")

        # Inizializzazione della profondita' della rete neurale
        """
            La profondita' della rete e' data dal numero di layer totali.
            Dato che 'l_sizes' tiene conto della dimensione dell'input e di tutte le dimensioni degli hidden layers, per contare la profondita' e' necessario togliere (input) e aggiungere (output) il valore 1.
        """
        self.depth = len(l_sizes) - 1 + 1

        # Inizializzazione degli hidden layers
        self.hidden_layers = []
        for i in range(1, len(l_sizes)):
            # print(f'Hidden layer n.{i}')
            prev_size = l_sizes[i-1]
            actual_size = l_sizes[i]
            hl = Layer(actual_size, prev_size)

            for j in range(len(hl.units)):
                # print(f'Neuron n.{j}')
                n = hl.units[j]
                n.weights = np.array(constants.STANDARD_DEVIATION * np.random.normal(size=prev_size))
                n.bias = constants.STANDARD_DEVIATION * np.random.normal()
                n.act_fun = l_act_funs[i]

            self.hidden_layers.append(hl)

        # Inizializzazione dell'output layer
        self.output_layer = Layer(output_size, l_sizes[-1])
        for j in range(len(self.output_layer.units)):
            n = self.output_layer.units[j]
            n.weights = np.array(constants.STANDARD_DEVIATION * np.random.normal(size=l_sizes[-1]))
            n.bias = constants.STANDARD_DEVIATION * np.random.normal()
            n.act_fun = l_act_funs[-1]

        # Inizializzazione della funzione di errore della rete
        self.err_fun = e_fun
        
        # Inizializzazione delle metriche di errore
        self.network_error = 0.0
        self.network_accuracy = 0.0
        self.average_error = 0.0

    # end
    
    # ####################################################################### #
    # METODI

    def load_input(self, x : list[float]) -> None:
        """
            Copia il vettore 'x' nel vettore 'inputs' della rete neurale.

            Parameters:
            -   x: il vettore di dati in input da caricare nella rete neurale

            Returns:
            -   None
        """

        if (len(x) > self.input_size):
            raise ValueError("La dimensione del vettore degli input non e' compatibile.")
        
        self.inputs = np.array(x)
            
    # end

    def forward_propagation(self,
                            x : list[float],
                            train : bool = False
    ) -> np.ndarray | list[np.ndarray]:
        
        """
            Calcola l'output complessivo della rete neurale, propagando i dati attraverso le connessioni di input dell'input layer, attraverso i calcoli intermedi degli hidden layers e, infine, attraverso l'ultimo strato dell'output layer.

            Parameters:
            -   train: serve a distinguere se l'applicazione del metodo è durante la fase di training o meno
            
            Returns:
            -   se train=False, un numpy.ndarray contenente l'output complessivo della rete.
            -   se train=True, una lista contenente gli output di tutti gli strati della rete.
        """

        out = []

        # Carica input nella rete neurale
        self.load_input(x)

        # Passa input al primo hidden layer
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
                         learning_rate : float = 0.00001
    ):
        
        """
            La backpropagation è l’algoritmo utilizzato durante la fase di addestramento delle reti neurali per determinare come un singolo campione del training set dovrebbe aggiustare i pesi ed i bias della rete, non solo in termini di aumento o diminuzione, ma in termini di qual è la proporzione relativa a tale cambiamento che possa causare la diminuzione più rapida possibile del valore della funzione di costo.

            Per capire quanto e' sensibile il valore della funzione di costo al cambiare dei pesi e dei bias dell'ultimo layer della rete, dobbiamo calcolarne la derivata prima parziale rispetto a questi pesi utilizzando iterativamente la regola della catena verso i layer precedenti, siccome questi aggiustamenti provocano una catena di effetti in tutta la rete.
            
            Parameters:
            -   x: ...
            -   y: ...
            -   learning_rate: ...

            Returns:
            -   numpy.ndarray : l'opposto del gradiente della funzione di costo.
        """

        # all_outputs = self.forward_propagation(x, train=True)
        # predicted_output = all_outputs[-1]
        # output_error = y - predicted_output
        # output_delta = output_error - auxfunc.sigmoid(predicted_output, der=True)

        # print(output_error)
        # TODO: da completare
        
    # end

    def resilient_back_propagation(self):
        """
            ...
            
            Parameters:
            -   ...: ...

            Returns:
            -   ...
        """
        
        pass
    # end

    def train(self):
        """
            ...
            
            Parameters:
            -   ...: ...

            Returns:
            -   ...
        """

        pass
    # end

    def predict(self):
        """
            ...
            
            Parameters:
            -   ...: ...

            Returns:
            -   ...
        """
        
        pass
    # end

    def compute_accuracy(self):
        """
            ...
            
            Parameters:
            -   ...: ...

            Returns:
            -   ...
        """
        
        pass
    # end

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe NeuralNetwork.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'NeuralNetwork(\n\tdepth = {self.depth},\n\tinput_size = {repr(self.input_size)},\n\tinputs = {pprint.pformat(self.inputs)},\n\thidden_layers = {pprint.pformat(self.hidden_layers)},\n\toutput_layer = {self.output_layer},\n\terr_fun = {self.err_fun},\n\tnetwork_error = {self.network_error},\n\tnetwork_accuracy = {self.network_accuracy},\n\taverage_error = {self.average_error}\n)'
    
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
# https://stackoverflow.com/questions/37835179/how-can-i-specify-the-function-type-in-my-type-hints
# https://docs.python.org/3/library/typing.html