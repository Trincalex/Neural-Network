'''

    artificial_neural_network.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di una rete neurale shallow feed-forward fully-connected (aka. Multilayer Perceptron) tramite paradigma di programmazione a oggetti.
    In particolare, la classe che implementa la rete neurale (NeuralNetwork) può essere composta di uno o più strati (Layer) che, a loro volta, possono essere composti di uno o più neuroni (Neuron).

'''

# ########################################################################### #
# LIBRERIE

from artificial_neuron import Neuron
from artificial_layer import Layer

import constants
import auxfunc
import numpy as np
import pprint
import time

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
            raise ValueError("La dimensione dell'input della rete neurale deve essere maggiore di 0.")
        
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
    def layers(self) -> list[Layer]:
        """È una lista di tutti i Layer della rete."""
        return self._layers
    # end

    @layers.setter
    def layers(self, value : list[Layer]) -> None:
        self._layers = value
    # end

    @property
    def weights(self) -> np.ndarray:
        """E' il vettore serializzato di tutti i pesi di tutti i neuroni della rete neurale. La sua dimensione e' pari al numero totale di pesi in ogni neurone della rete."""
        return np.array([w_elem for l in self.layers for w_vet in l.weights for w_elem in w_vet])
    # end

    @weights.setter
    def weights(self, value : np.ndarray) -> None:
        # print("Neural Network:", len(value), len(self.weights))
        if (len(value) <= 0 or len(value) > len(self.weights)):
            raise ValueError("La dimensione del vettore dei pesi non e' compatibile.")

        start = 0
        end = 0
        for l in self.layers:
            end += l.weights.size
            l.weights = value[start:end]
            start = end

    # end

    @property
    def biases(self) -> np.ndarray:
        """E' il vettore serializzato di tutti i bias di tutti i neuroni della rete neurale. La sua dimensione e' pari al numero totale di neuroni nei layer della rete."""
        return np.array([b_elem for l in self.layers for b_vet in l.biases for b_elem in b_vet])
    # end

    @biases.setter
    def biases(self, value : np.ndarray) -> None:
        # print("Neural Network:", len(value), len(self.biases))
        if (len(value) <= 0 or len(value) > len(self.biases)):
            raise ValueError("La dimensione del vettore dei bias non e' compatibile.")
        
        start = 0
        end = 0
        for l in self.layers:
            end += l.layer_size
            l.biases = value[start:end]
            start = end

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
    def training_error(self):
        """..."""
        return self._training_error
    # end

    @training_error.setter
    def training_error(self, value) -> None:
        self._training_error = value
    # end

    @property
    def validation_error(self):
        """..."""
        return self._validation_error
    # end

    @validation_error.setter
    def validation_error(self, value):
        self._validation_error = value
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
            e_fun : constants.ErrorFunctionType = auxfunc.sum_of_squares
    ) -> None:
        
        """
            È il costruttore della classe NeuralNetwork.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   i_size : è la dimensione del vettore in input alla rete neurale
            -   hidden_sizes : può essere un numero o una lista contenente la dimensione di uno o più-  hidden layer della rete neurale.
            -   output_size : è la dimensione dell'output layer della rete neurale.
            -   hidden_act_funs : può essere una funzione o una lista contenente le funzioni di attivazione di uno o più hidden layer della rete neurale.
            -   output_act_fun : è la funzione di attivazione dei neuroni dell'output layer.
            -   e_fun : è la funzione di errore utilizzata per verificare la qualità della rete neurale.

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
        
        rng = np.random.default_rng(0)

        # Inizializzazione degli hidden layers
        self.layers = []
        for i in range(1, len(l_sizes)):
            # print(f'Hidden layer n.{i}')
            prev_size = l_sizes[i-1]
            actual_size = l_sizes[i]
            hl = Layer(actual_size, prev_size, l_act_funs[i])

            for j in range(len(hl.units)):
                # print(f'Neuron n.{j}')
                n = hl.units[j]
                n.weights = rng.random(prev_size)

            for j in range(len(hl.units)):
                n = hl.units[j]
                n.bias = rng.random()

                # n.weights = np.random.normal(scale=constants.STANDARD_DEVIATION, size=prev_size)
                # n.bias = np.random.normal(scale=constants.STANDARD_DEVIATION)

            self.layers.append(hl)

        # Inizializzazione dell'output layer
        ol = Layer(output_size, l_sizes[-1], l_act_funs[-1])
        for j in range(len(ol.units)):
            n = ol.units[j]
            n.weights = rng.random(l_sizes[-1])

        for j in range(len(ol.units)):
            n = ol.units[j]
            n.bias = rng.random()

            # n.weights = np.random.normal(scale=constants.STANDARD_DEVIATION, size=l_sizes[-1])
            # n.bias = np.random.normal(scale=constants.STANDARD_DEVIATION)

        self.layers.append(ol)

        # Inizializzazione della profondita' della rete neurale
        # La profondita' della rete e' data dal numero di layer totali.
        self.depth = len(self.layers)

        # Inizializzazione della funzione di errore della rete
        self.err_fun = e_fun
        
        # Inizializzazione delle metriche di errore
        self.training_error = 0.0
        self.validation_error = 0.0
        self.network_accuracy = 0.0

    # end
    
    # ####################################################################### #
    # METODI PRIVATI

    def __load_input(self, x : list[float]) -> None:
        """
            Copia il vettore 'x' nel vettore 'inputs' della rete neurale.

            Parameters:
            -   x : il vettore di dati in input da caricare nella rete neurale.

            Returns:
            -   None
        """

        if (len(x) > self.input_size):
            raise ValueError("La dimensione del vettore degli input non e' compatibile.")
        
        if (not isinstance(x, np.ndarray)):
            self.inputs = np.array(x)
        else:
            self.inputs = x
            
    # end

    def forward_propagation(
            self,
            x : list[float],
            train : bool = False
    ) -> np.ndarray | tuple[list[np.ndarray], list[np.ndarray]]:
        
        """
            Calcola l'output complessivo della rete neurale, propagando i dati attraverso le connessioni di input dell'input layer, attraverso i calcoli intermedi degli hidden layers e, infine, attraverso l'ultimo strato dell'output layer.

            Parameters:
            -   x : il vettore di dati in input.
            -   train : serve a distinguere se l'applicazione del metodo è per la fase di training o meno.
            
            Returns:
            -   se train=False, un numpy.ndarray contenente i valori di attivazione complessivi della rete, cioe' i valori di attivazione dell'output layer.
            -   se train=True, una prima lista contenente gli input pesati di ogni layer della rete ed una seconda lista contenente i valori di attivazione di ogni layer della rete.
        """

        outputs = []
        activations = []

        # Carica input nella rete neurale
        self.__load_input(x)

        # Passa input al primo hidden layer
        activations.append(self.inputs)
        # pprint.pprint(activations[-1])

        for i in range(self.depth):
            l = self.layers[i]

            # Aggiorna input dell'i-esimo layer
            for j in range(len(l.units)):
                n = l.units[j]
                n.inputs = activations[-1]
            
            # Calcola output dell'i-esimo layer
            if train:
                out, act = l.activate(train=True)
                outputs.append(out)
            else:
                act = l.activate(train=False)
                
            activations.append(act)
            # pprint.pprint(activations[-1])
        
        activations.pop(0)

        if train:
            return outputs, activations

        return activations[-1]
    
    # end

    def __compute_delta_cost_activation(
        self,
        layer_index : int,
        neuron_index : int,
        network_outputs : list[np.ndarray],
        network_activations : list[np.ndarray],
        target : list[float]
    ) -> float:

        """
            ...

            Parameters:
            -   ... : ...

            Returns:
            -   ... : ...
        
        """


    
    # end

    def __back_propagation(
        self,
        network_outputs : list[np.ndarray],
        network_activations : list[np.ndarray],
        target_label : list[float],
        learning_rate : float = constants.DEFAULT_LEARNING_RATE
    ) -> np.ndarray:
        
        """
            Aggiusta i valori dei pesi ed i bias della rete per diminuire il valore della funzione di costo rispetto all'esempio di training e la corrispondente etichetta in input. Calcola la derivata prima parziale (gradiente) della funzione di costo rispetto a tutti i pesi della rete utilizzando iterativamente la regola della catena verso i layer precedenti (essendo la rete fully-connected, l'aggiustamento dei pesi di un layer provoca una catena di effetti in tutti i layer successivi).
            
            Parameters:
            -   network_outputs : la lista di output di ogni layer della rete.
            -   network_activations : la lista di valori di attivazione di ogni layer della rete.
            -   target_label : l'etichetta dell'esempio di training
            -   learning_rate : e' un parametro utilizzato per l'aggiornamento dei pesi che indica quanto i pesi debbano essere modificati in risposta all'errore calcolato.

            Returns:
            -   np.ndarray : il gradiente della funzione di costo rispetto ai pesi della rete neurale.
            -   np.ndarray : il gradiente della funzione di costo rispetto ai bias della rete neurale.
        """


        
    # end

    def __resilient_back_propagation(self):
        """
            ...
            
            Parameters:
            -   ... : ...

            Returns:
            -   ... : ...
        """
        
        pass

    # end

    def __compute_accuracy(predictions : np.ndarray, targets : np.ndarray) -> float:
        """
            Calcola l'accuratezza della rete neurale confrontando le previsioni con i target corrispondenti.

            Parameters:
            -   predictions : e' un'array contenente tutte le previsioni della rete.
            -   targets : e' un'array contenente le etichette vere corrispondenti alle previsioni (ground truth).

            Returns:
            -   float : il rapporto tra predizioni corrette e totale delle predizioni.
        """

        # if (not predictions.shape == targets.shape):
        #     raise ValueError("...")

        pass

    # end
    
    # ####################################################################### #
    # METODI PUBBLICI

    def train(self,
              training_data : np.ndarray,
              training_labels : np.ndarray,
              validation_data : np.ndarray,
              validation_labels : np.ndarray,
              epochs : int = constants.DEFAULT_EPOCHS,
              learning_rate : float = constants.DEFAULT_LEARNING_RATE
    ):
        
        """
            Addestra la rete neurale tramite il training set ed il validation set dati in input.
            Il processo di addestramento ripete le fasi di forward propagation, backpropagation (con calcolo della funzione di costo) e il conseguente aggiornamento dei pesi per un numero limitato di iterazioni (epochs).
            
            Parameters:
            -   training_data : una matrice numpy.ndarray contenente i dati di input per l'addestramento. Ogni riga rappresenta un esempio di addestramento.
            -   training_labels : una matrice numpy.ndarray contenente le etichette corrispondenti per i dati di addestramento. Ogni riga rappresenta l'etichetta per il rispettivo esempio di addestramento.
            -   validation_data : una matrice numpy.ndarray da utilizzare per la fase di validazione dell'addestramento. Ogni riga rappresenta un esempio di addestramento.
            -   validation_labels : una matrice numpy.ndarray da utilizzare per la fase di validazione dell'addestramento. Ogni riga rappresenta l'etichetta per il rispettivo esempio di addestramento.
            -   epochs : il numero di iterazioni per cui il modello deve essere addestrato. Un'epoca e' un'esecuzione completa dell'addestramento attraverso l'intero training_set.
            -   learning_rate : e' un parametro utilizzato per l'aggiornamento dei pesi che indica quanto i pesi debbano essere modificati in risposta all'errore calcolato.

            Returns:
            -   ... : ...
        """

        # Controllo sulla compatibilita' di training_data e training_labels
        if (not training_data.shape[0] == training_labels.shape[0]):
            raise constants.TrainError(f"Le dimensioni del dataset [{training_data.shape[0]}] e delle labels [{training_labels.shape[0]}] di addestramento non sono compatibili.")

        # Controllo sulla compatibilita' di validation_data e validation_labels
        if (not validation_data.shape[0] == validation_labels.shape[0]):
            raise constants.TrainError(f"Le dimensioni del dataset [{validation_data.shape[0]}] e delle labels [{validation_labels.shape[0]}] per la validazione non sono compatibili.")

        training_errors = []
        training_weights = []

        validation_errors = []

        data = 0
        label = 1

        start_time = time.time()
        print("Addestramento in corso...")

        for e in range(epochs):

            # TRAINING
            for n, example in enumerate(zip(training_data, training_labels)):

                # print(f"Esempio n.{n+1}")
                # print(f"\tExample: {example[data]}\n\tLabel: {example[label]}\n")

                # STEP 1: forward propagation
                training_outputs, training_activations = self.__forward_propagation(
                    example[data],
                    train=True
                )

                # print("training_outputs\n")
                # pprint.pprint(training_outputs)
                # print("training_activations\n")
                # pprint.pprint(training_activations)

                # STEP 2: calcolo dell'errore di training
                training_errors.append(self.err_fun(training_activations[-1], example[label]))

                # STEP 3: backpropagation
                gradient_weights, gradient_biases = self.__back_propagation(
                    training_outputs,
                    training_activations,
                    example[label]
                )

                # STEP 4: aggiornamento dei pesi
                training_weights.append({
                    "Weights" : np.array([
                        w - learning_rate * g
                        for w, g in zip(
                            self.weights,
                            gradient_weights
                        )
                    ]),
                    "Biases" : np.array([
                        b - learning_rate * g
                        for b, g in zip(
                            self.biases,
                            gradient_biases
                        )
                    ])
                })

                self.weights = training_weights[-1]["Weights"]
                self.biases = training_weights[-1]["Biases"]

            # end for n, example

            # VALIDATION
            for n, example in enumerate(zip(validation_data, validation_labels)):

                print(f"Esempio n.{n+1}\n")
                # print(f"\tExample: {example[data]}\n\tLabel: {example[label]}\n")

                # STEP 1: forward propagation
                validation_prediction = self.__forward_propagation(example[data])

                # STEP 2: calcolo dell'errore di validation
                validation_errors.append(self.err_fun(validation_prediction, example[label]))
            
            # end for n, example

            end_time = time.time()
            tot_time = end_time - start_time

            if (e == 0 or (e+1) % (epochs / constants.DEFAULT_EPOCHS) == 0):
                print(f"\nEpoca {e+1} di {epochs}")
                print(f"\tTempo totale: {round(tot_time, 3)} secondi")
                print(f"\tErrore di addestramento: {round(training_errors[-1], 5)}")
                print(f"\tErrore di validazione: {round(validation_errors[-1], 5)}")
                print(f"\tAccuracy: ")

        # end for e

        print(f"L'addestramento ha impiegato {round(tot_time, 3)} secondi.")
        print()

        # Scelta dei parametri corrispondenti alla miglior rete (errore di validazione minimo)
        best_net = int(np.argmin(validation_errors, keepdims=False))
        # print(best_net, np.min(validation_errors))
        self.weights = training_weights[best_net]["Weights"]
        self.biases = training_weights[best_net]["Biases"]

    # end

    def predict(self, x : list[float]) -> int:
        """
            Calcola una predizione per l'input dato in base alla configurazione attuale di pesi e bias della rete neurale. Inoltre, visualizza nel terminale le probabilita' delle predizioni di tutto l'output layer utilizzando la funzione "auxfunc.softmax()".
            
            Parameters:
            -   x : il vettore di dati in input.

            Returns:
            -   label : l'indice del neurone nell'output layer che ottiene il valore di attivazione piu' alto.
        """

        prediction = self.__forward_propagation(x)
        label = np.argmax(prediction)

        print(f"Predizione della rete: {constants.ETICHETTE_CLASSI[label]}")
        for i, pred in enumerate(prediction):
            print(f'\tClasse {i}: {round(pred, 5)}')
        print()

        # Utilizza la funzione softmax per ottenere valori probabilistici della predizione
        probabilities = auxfunc.softmax(prediction)
        print("Probabilità della predizione:")
        for i, prob in enumerate(probabilities):
            print(f'\tClasse {i}: {round(prob * 100, 5)}')
        print()

        return label
        
    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe NeuralNetwork.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'NeuralNetwork(\n\tdepth = {self.depth},\n\tinput_size = {repr(self.input_size)},\n\tinputs = {pprint.pformat(self.inputs)},\n\tnetwork_layers = {pprint.pformat(self.layers)},\n\terr_fun = {self.err_fun},\n\ttraining_error = {self.training_error},\n\tvalidation_error = {self.validation_error},\n\tnetwork_accuracy = {self.network_accuracy}\n)'
    
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
# https://towardsdatascience.com/understanding-backpropagation-algorithm-7bb3aa2f95fd
# http://neuralnetworksanddeeplearning.com/chap2.html