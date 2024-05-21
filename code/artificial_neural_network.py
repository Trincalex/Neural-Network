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
                n.weights = np.array(constants.STANDARD_DEVIATION * np.random.normal(size=prev_size))
                n.bias = constants.STANDARD_DEVIATION * np.random.normal()

            self.layers.append(hl)

        # Inizializzazione dell'output layer
        ol = Layer(output_size, l_sizes[-1], l_act_funs[-1])
        for j in range(len(ol.units)):
            n = ol.units[j]
            n.weights = np.array(constants.STANDARD_DEVIATION * np.random.normal(size=l_sizes[-1]))
            n.bias = constants.STANDARD_DEVIATION * np.random.normal()

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

    def __forward_propagation(
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
            -   se train=True, una prima lista contenente i valori degli output intermedi di ogni layer della rete ed una seconda lista contenente i valori di attivazione di ogni layer della rete.
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

        if layer_index < 0 or layer_index >= self.depth:
            raise ValueError("L'indice inserito per il layer non e' valido.")
        
        if neuron_index < 0 or neuron_index >= self.layers[layer_index].layer_size:
            raise ValueError("L'indice inserito per il neurone non e' valido.")
        
        if layer_index == self.depth-1:

            out = self.err_fun(network_activations[layer_index], target, der=True)
            # print("cost_gradient:", round(out[neuron_index], 3))
            # print()
            return out[neuron_index]
        
        else:

            out = 0
            next_size = self.layers[layer_index+1].layer_size

            for j in range(next_size):
                # print("layer_index+1:", layer_index+1, "layer_size:", next_size)
                # print("next_neuron:", j, "prev_neuron:", neuron_index)

                weights_size = self.layers[layer_index+1].weights.shape[1]
                weight_index = (layer_index+1) * (next_size * weights_size) + j * weights_size + neuron_index
                delta_za = self.weights[weight_index]

                delta_az = self.layers[layer_index+1].act_fun(network_outputs[layer_index+1][j], der=True)

                delta_Ca = self.__compute_delta_cost_activation(
                    layer_index+1, j,
                    network_outputs,
                    network_activations,
                    target
                )

                # print("delta_za:", round(delta_za, 3), "; delta_az:", round(delta_az, 3), "; delta_Ca:", round(delta_Ca, 3))

                delta = delta_za * delta_az * delta_Ca
                out += delta
            
            # end for j

            # print("derivative_cost_activation:", round(out, 3))
            # print()
            return out
    
    # end

    def __back_propagation(
            self,
            network_outputs : list[np.ndarray],
            network_activations : list[np.ndarray],
            target_label : list[float],
            learning_rate : float = constants.LEARNING_RATE
    ) -> np.ndarray:
        
        """
            Aggiusta i valori dei pesi ed i bias della rete per diminuire il valore della funzione di costo rispetto all'esempio di training e la corrispondente etichetta in input. Calcola la derivata prima parziale (gradiente) del valore della funzione di costo rispetto a tutti i pesi della rete utilizzando iterativamente la regola della catena verso i layer precedenti (essendo la rete fully-connected, l'aggiustamento dei pesi di un layer provoca una catena di effetti in tutti i layer successivi).
            
            Parameters:
            -   network_outputs : la lista di output di ogni layer della rete.
            -   network_activations : la lista di valori di attivazione di ogni layer della rete.
            -   target_label : l'etichetta dell'esempio di training
            -   learning_rate : e' un parametro utilizzato per l'aggiornamento dei pesi che indica quanto i pesi debbano essere modificati in risposta all'errore calcolato.

            Returns:
            -   np.ndarray : il gradiente della funzione di costo rispetto ai pesi della rete neurale.
            -   np.ndarray : il gradiente della funzione di costo rispetto ai bias della rete neurale.
        """

        gradient_weights = []
        gradient_biases = []
        gradient_size = 0

        for l in reversed(range(self.depth)):
            # print("Layer:", l)

            curr_size = self.layers[l].layer_size
            prev_size = self.input_size if l == 0 else self.layers[l-1].layer_size
            # print("curr_size: ", curr_size, "prev_size: ", prev_size)

            for j in range(curr_size):

                delta_az = self.layers[l].act_fun(network_outputs[l][j], der=True)
                delta_Ca = self.__compute_delta_cost_activation(
                    l, j,
                    network_outputs,
                    network_activations,
                    target_label
                )

                for k in range(prev_size):

                    delta_zw = self.inputs[k] if l == 0 else network_activations[l-1][k]
                    delta = delta_zw * delta_az * delta_Ca

                    # print(f"Componente n.{gradient_size}")
                    # print("\tdelta_zw:", round(delta_zw, 3))
                    # print("\tdelta_az:", round(delta_az, 3))
                    # print("\tdelta_Ca:", round(delta_Ca, 3), "\n")

                    gradient_weights.append(delta)

                    # Conteggio posizioni del gradiente per i pesi della rete
                    gradient_size += 1

                # end for k
            # end for j
        # end for l

        for l in reversed(range(self.depth)):
            curr_size = self.layers[l].layer_size

            for j in range(curr_size):

                delta_zb = 1
                delta = delta_zb * delta_az * delta_Ca

                # print(f"Componente n.{gradient_size}")
                # print("\tdelta_zb:", round(delta_zb, 3))
                # print("\tdelta_az:", round(delta_az, 3))
                # print("\tdelta_Ca:", round(delta_Ca, 3), "\n")

                gradient_biases.append(delta)
                
                # Conteggio posizioni del gradiente per i bias della rete
                gradient_size += 1
            
            # end for j
        # end for l

        # pprint.pprint(gradient_weights)
        # pprint.pprint(gradient_biases)
        # print("weights:", len(self.weights))
        # print("biases:", len(self.biases))
        # print("all_weights:", len(np.concatenate((self.weights, self.biases))))
        # print("gradient_size:", gradient_size)

        return gradient_weights, gradient_biases

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
            -   targets : e' un'array contenente i target reali corrispondenti alle previsioni.

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
              epochs : int = 35,
              learning_rate : float = constants.LEARNING_RATE
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

        l = []
        start_time = time.time()

        for e in range(epochs):

            for n, example in enumerate(zip(training_data, training_labels)):

                data = 0
                label = 1

                # print(f"Esempio n.{n+1}\n")
                # print(f"\tExample: {example[data]}\n\tLabel: {example[label]}\n")

                # STEP 1: forward propagation
                network_outputs, network_activations = self.__forward_propagation(
                    example[data],
                    train=True
                )

                # print("network_outputs\n")
                # pprint.pprint(network_outputs)
                # print("network_activations\n")
                # pprint.pprint(network_activations)

                # STEP 2: backpropagation
                gradient_weights, gradient_biases = self.__back_propagation(
                    network_outputs,
                    network_activations,
                    example[label]
                )

                # STEP 3: aggiornamento dei pesi
                self.weights = np.array(
                    [
                        w - learning_rate * g
                        for w, g in zip(
                            self.weights,
                            gradient_weights
                        )
                    ]
                )
                
                self.biases = np.array(
                    [
                        b - learning_rate * g
                        for b, g in zip(
                            self.biases,
                            gradient_biases
                        )
                    ]
                )

            # end for n, example

            # TODO: compute_accuracy

            end_time = time.time()
            tot_time = end_time - start_time
            if (e == 0 or (e+1) % (epochs / constants.NUM_STAMPE_EPOCHE) == 0):
                print(f"Epoca n.{e+1} terminata in {round(tot_time, 3)} secondi.")
            print()

        # end for e

        print(f"L'addestramento ha impiegato {round(tot_time, 3)} secondi.")
        print()

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

        print("Predizione della rete:")
        print(f'\t{constants.ETICHETTE_CLASSI[label]}')
        print()

        # Utilizza la funzione softmax per ottenere valori probabilistici della predizione
        probabilities = auxfunc.softmax(prediction)
        print("Probabilità delle predizioni:")
        for p in [f'\tClasse {i}: {prob}' for i, prob in enumerate(probabilities)]:
            print(p)
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