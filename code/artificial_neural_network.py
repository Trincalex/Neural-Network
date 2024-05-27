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
        
        if (not value.size == self.input_size):
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
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore dei pesi deve essere di tipo 'numpy.ndarray'.")
        
        # print("Neural Network:", value.size, self.weights.size
        if (not value.size == self.weights.size):
            raise ValueError("Il vettore dei pesi non e' compatibile con questa rete neurale.")

        start = 0
        end = 0
        for l in self.layers:
            end += l.weights.shape[0] * l.weights.shape[1]
            # l.weights = value[start:end]
            l.weights = np.reshape(value[start:end], l.weights.shape)
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
            l.biases = np.reshape(value[start:end], l.biases.shape)
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

    @property
    def debug(self) -> bool:
        """..."""
        return self._debug

    @debug.setter
    def debug(self, value : bool) -> None:
        self._debug = value

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            i_size : int,
            hidden_sizes : list[int],
            output_size : int,
            hidden_act_funs : list[constants.ActivationFunctionType] = auxfunc.sigmoid,
            output_act_fun : constants.ActivationFunctionType = auxfunc.sigmoid,
            e_fun : constants.ErrorFunctionType = auxfunc.sum_of_squares,
            random_init : bool = True,
            debug : bool = False
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
            -   random_init : indica se pesi e bias della rete neurale saranno inizializzati tramite un generatore di valori casuali con seed fissato o meno.
            -   debug : consente di attivare la modalita' di debug per stampare in console i valori attuali delle strutture dati coinvolte nell'addestramento della rete neurale.

            Returns:
            -   None
        """

        # Scelta della modalita' di esecuzione
        self.debug = debug
        
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
        
        if not random_init:
            rng = np.random.default_rng(constants.DEFAULT_RANDOM_SEED)

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
                if random_init:
                    n.weights = np.random.normal(loc=0.0, scale=constants.STANDARD_DEVIATION, size=prev_size)
                else:
                    n.weights = rng.normal(loc=0.0, scale=constants.STANDARD_DEVIATION, size=prev_size)

            for j in range(len(hl.units)):
                n = hl.units[j]
                if random_init:
                    n.bias = np.random.normal(loc=0.0, scale=constants.STANDARD_DEVIATION)
                else:
                    n.bias = rng.normal(loc=0.0, scale=constants.STANDARD_DEVIATION)

            self.layers.append(hl)

        # Inizializzazione dell'output layer
        ol = Layer(output_size, l_sizes[-1], l_act_funs[-1])
        for j in range(len(ol.units)):
            n = ol.units[j]
            if random_init:
                n.weights = np.random.normal(loc=0.0, scale=constants.STANDARD_DEVIATION, size=l_sizes[-1])
            else:
                n.weights = rng.normal(loc=0.0, scale=constants.STANDARD_DEVIATION, size=l_sizes[-1])

        for j in range(len(ol.units)):
            n = ol.units[j]
            if random_init:
                n.bias = np.random.normal(loc=0.0, scale=constants.STANDARD_DEVIATION)
            else:
                n.bias = rng.normal(loc=0.0, scale=constants.STANDARD_DEVIATION)

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

    def __delta_output_layer(
            self,
            output_layer_outputs : np.ndarray,
            output_layer_activations : np.ndarray,
            target : np.ndarray
    ) -> np.ndarray:
        
        """
            Calcola il vettore le cui componenti sono le derivate prime parziali della funzione di costo della rete neurale rispetto agli input pesati dell'output layer.
            E' l'implementazione dell'equazione (BP1a) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.

            Parameters:
            -   output_layer_outputs : e' il vettore di input pesati dell'output layer.
            -   output_layer_activations : e' il vettore di valori di attivazione dell'output layer.
            -   target : e' l'etichetta di una determinata coppia del dataset.

            Returns:
            -   np.ndarray : il gradiente della funzione di costo rispetto agli input pesati dell'output layer.
        """

        """
            E' un vettore le cui componenti sono le singole derivate parziali della funzione di errore rispetto al singolo valore di attivazione dell'output layer.
            Esprime quanto cambia la funzione di costo rispetto a questi valori di attivazione.
        """

        delta_Ca = self.err_fun(output_layer_activations, target, der=True)

        """
            I pesi nell'output layer si addestrano lentamente se il valore di attivazione calcolato su 'out' e' molto basso o molto alto (per la sigmoide, ad esempio, vicino allo 0 o vicino a 1, rispettivamente).
            In questo caso, la derivata prima restituisce un valore molto vicino allo 0. Si dice che il neurone dell'output layer si e' saturato e, di conseguenza, il peso non si addestra piu' (o si addestra lentamente). Lo stesso vale anche per i bias della rete neurale.
            Una possibile soluzione per prevenire il rallentamento dell'apprendimento, ad esempio, potrebbe essere quella di scegliere una funzione di attivazione la cui derivata e' sempre positiva e che non si avvicina mai allo 0.
        """
        
        delta_az = np.array([
            self.layers[-1].act_fun(out, der=True)
            for out in output_layer_outputs
        ])

        return np.multiply(delta_Ca, delta_az)

    # end

    def __delta_layer(
            self,
            network_outputs : list[np.ndarray],
            network_activations : list[np.ndarray],
            network_weights : np.ndarray,
            delta_output_layer : np.ndarray,
            layer_index: int
    )-> np.ndarray:
        
        """
            Calcola il vettore le cui componenti sono le derivate prime parziali della funzione di costo della rete neurale rispetto agli input pesati di un layer.
            E' l'implementazione dell'equazione (BP2) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.

            Parameters:
            -   network_outputs : la lista di output di ogni layer della rete.
            -   network_activations : la lista di valori di attivazione di ogni layer della rete.
            -   delta_output_layer : il gradiente della funzione di costo rispetto agli input pesati dell'output layer.
            -   layer_index : e' l'indice del layer scelto che contiene il neurone indicato da 'neuron_index' (corrisponde a 'l' nell'equazione proposta da Nielsen).

            Returns:
            -   np.ndarray : il gradiente della funzione di costo rispetto agli input pesati del layer scelto nella rete neurale.
        """
        
        if layer_index == self.depth-1:
            return delta_output_layer
        
        if layer_index+1 == self.depth-1:
            delta_tmp = delta_output_layer
        else:
            delta_tmp = self.__delta_layer(
                network_outputs,
                network_activations,
                network_weights,
                delta_output_layer,
                layer_index+1
            )

        layer = self.layers[layer_index]
        next_layer = self.layers[layer_index+1]

        start = layer.layer_size * layer.units[0].neuron_size
        end = start + next_layer.layer_size * next_layer.units[0].neuron_size
        weights_shape = (next_layer.layer_size, layer.layer_size)
        next_layer_weights = np.reshape(network_weights[start:end], weights_shape).T

        delta_Ca = np.dot(next_layer_weights, delta_tmp)
        delta_az = np.array([
            layer.act_fun(out, der=True)
            for out in network_outputs[layer_index]
        ])

        # Restituisce un vettore le cui componenti sono piccole se i corrispondenti neuroni sono vicini alla saturazione. In generale, qualsiasi input pesato di un neurone pesato si addestra lentamente (tranne nei casi in cui il vettore dei pesi può compensare questi valori piccoli).
        return np.multiply(delta_Ca, delta_az)

    # end

    def __back_propagation(
            self,
            network_outputs : list[np.ndarray],
            network_activations : list[np.ndarray],
            network_weights : np.ndarray,
            target : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        """
            Aggiusta i valori dei pesi ed i bias della rete per diminuire il valore della funzione di costo rispetto all'esempio di training e la corrispondente etichetta in input.
            Calcola il gradiente della funzione di costo rispetto a tutti i pesi della rete utilizzando le quattro equazioni (BP1)-(BP4) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen. Tali equazioni sono una diretta conseguenza della regola della catena del calcolo multivariabile (essendo la rete fully-connected, l'aggiustamento dei pesi di un layer provoca una catena di effetti in tutti i layer successivi).
            
            Parameters:
            -   network_outputs : la lista di output di ogni layer della rete.
            -   network_activations : la lista di valori di attivazione di ogni layer della rete.
            -   network_weights : ...
            -   target : e' l'etichetta di una determinata coppia del dataset.

            Returns:
            -   np.ndarray : il gradiente della funzione di costo rispetto ai pesi della rete neurale.
            -   np.ndarray : il gradiente della funzione di costo rispetto ai bias della rete neurale.
        """

        gradient_weights = []
        gradient_biases = []

        """
            STEP 1:
            Calcolo dell'errore sull'output layer.
        """

        delta_output_layer = self.__delta_output_layer(
            network_outputs[-1],
            network_activations[-1],
            target
        )

        if self.debug:
            with np.printoptions(threshold=np.inf):
                print("--- BACKPROPAGATION (delta_output_layer) ---\n")
                pprint.pprint(delta_output_layer)
                print("\n-----")

        for l in range(self.depth):
            # print("layer_index:", l)

            """
                STEP 2:
                Retro-propagazione dell'errore ai layer precedenti.
            """

            delta_layer = self.__delta_layer(
                network_outputs,
                network_activations,
                network_weights,
                delta_output_layer,
                l
            )

            if self.debug:
                with np.printoptions(threshold=np.inf):
                    print(f"--- BACKPROPAGATION (delta_layer_{l}) ---\n")
                    pprint.pprint(delta_layer)
                    print("\n-----")

            """
                STEP 3:
                Calcolo del gradiente della funzione di costo rispetto ai bias della rete neurale, cioe' come cambia la funzione di costo rispetto ai bias di tutti i neuroni della rete neurale.
                E' l'implementazione dell'equazione (BP3) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.
            """

            curr_size = self.layers[l].layer_size

            for j in range(curr_size):
                gradient_biases.append(delta_layer[j])

            """
                STEP 4:
                Calcolo del gradiente della funzione di costo rispetto ai pesi della rete neurale, cioe' come cambia la funzione di costo rispetto al peso di tutte le connessioni tra due neuroni di due layer adiacenti della rete neurale.
                E' l'implementazione dell'equazione (BP4) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.
            """

            prev_layer_activations = self.inputs if l == 0 else network_activations[l-1]
            prev_size = self.input_size if l == 0 else self.layers[l-1].layer_size

            for j in range(curr_size):
                for k in range(prev_size):
                    gradient_weights.append(prev_layer_activations[k] * delta_layer[j])

            if self.debug:
                hIn = 1
                oIn = 8
                if l == 1:
                    actual_delta_Ca = 2 * (target[oIn] - network_activations[-1][oIn])
                    test_delta_Ca = self.err_fun(network_activations[-1], target, der=True)[oIn]
                    print("delta_Ca:", actual_delta_Ca, test_delta_Ca)

                    actual_delta_az = network_activations[-1][oIn]*(1-network_activations[-1][oIn])
                    test_delta_az = self.layers[-1].act_fun(network_outputs[-1][oIn], der=True)
                    print("delta_az:", actual_delta_az, test_delta_az)

                    actual_delta_zw = network_activations[-2][hIn]
                    print("delta_zw:", actual_delta_zw, prev_layer_activations[hIn])

                    print(
                        "\n\n",
                        "actual_delta_layer:", actual_delta_Ca * actual_delta_az,
                        "test_delta_layer:", delta_layer[oIn], delta_output_layer[oIn]
                    )

                    # for index, element in enumerate(gradient_weights[784*32:]):
                    #     if element == (actual_delta_Ca * actual_delta_az * actual_delta_zw):
                    #         print("gradient_weights:", index)

                    print(
                        "actual:", actual_delta_Ca * actual_delta_az * actual_delta_zw,
                        "test:", gradient_weights[784*3+3*oIn+hIn]
                    )

        return np.array(gradient_weights), np.array(gradient_biases)
        
    # end

    def __update_rule(
            self,
            training_data : np.ndarray,
            network_weights : np.ndarray,
            network_biases : np.ndarray,
            training_weights : list[np.ndarray],
            training_biases : list[np.ndarray],
            learning_rate : float
    ):
        
        """
            ...
            
            Parameters:
            -   ... : ...

            Returns:
            -   ... : ...
            -   learning_rate : e' un parametro utilizzato per l'aggiornamento dei pesi che indica quanto i pesi debbano essere modificati in risposta all'errore calcolato.
        """

        # Le righe sono il numero di esempi di addestramento.
        rows_size = len(training_data)
        # Le colonne sono il numero di pesi e/o bias della rete neurale.
        cols_size = -1

        gradient_weights = np.mean(
            np.reshape(
                training_weights,
                (rows_size, cols_size)
            ),
            axis=0
        )

        gradient_biases = np.mean(
            np.reshape(
                training_biases,
                (rows_size, cols_size)
            ),
            axis=0
        )

        if self.debug:
            with np.printoptions(threshold=np.inf):
                print("--- GRADIENT_WEIGHTS (mean) ---\n")
                pprint.pprint(gradient_weights)
                print("\n-----")
                print("--- GRADIENT_BIASES (mean) ---\n")
                pprint.pprint(gradient_biases)
                print("\n-----")
                print("--- NETWORK WEIGHTS (end) ---\n")
                pprint.pprint(network_weights)
                print("\n-----")
                print("--- NETWORK BIASES (end) ---\n")
                pprint.pprint(network_biases)
                print("\n-----")

        return {
            "Weights" : np.array([
                w - learning_rate * g
                for w, g in zip(
                    network_weights,
                    gradient_weights
                )
            ]),
            "Biases" : np.array([
                b - learning_rate * g
                for b, g in zip(
                    network_biases,
                    gradient_biases
                )
            ])
        }

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

    def __compute_error(self, predictions : np.ndarray, targets : np.ndarray) -> float:
        """
            Calcola l'errore della rete neurale confrontando le previsioni con i target corrispondenti.

            Parameters:
            -   predictions : e' un'array contenente tutte le previsioni della rete.
            -   targets : e' un'array contenente le etichette vere corrispondenti alle previsioni (ground truth).

            Returns:
            -   float : il rapporto tra predizioni corrette e totale delle predizioni.
        """

        # print(predictions.shape, targets.shape)
        if not predictions.shape == targets.shape:
            raise constants.TrainError("Il numero di predizioni e di etichette non sono compatibili.")
        
        errors = [self.err_fun(x, y) for x, y in zip(predictions, targets)]
        return np.mean(errors)

    # end
    
    def __compute_accuracy(self, predictions : np.ndarray, targets : np.ndarray) -> float:
        """
            Calcola l'accuratezza della rete neurale confrontando le previsioni con i target corrispondenti.

            Parameters:
            -   predictions : e' un'array contenente tutte le previsioni della rete.
            -   targets : e' un'array contenente le etichette vere corrispondenti alle previsioni (ground truth).

            Returns:
            -   float : il rapporto tra predizioni corrette e totale delle predizioni.
        """

        # print(predictions.shape, targets.shape)
        if not predictions.shape == targets.shape:
            raise constants.TrainError("Il numero di predizioni e di etichette non sono compatibili.")

        matches = [int(np.argmax(x) == np.argmax(y)) for x, y in zip(predictions, targets)]
        return np.sum(matches), np.sum(matches) / len(predictions) * 100

    # end
    
    # ####################################################################### #
    # METODI PUBBLICI

    def train(
            self,
            training_data : np.ndarray,
            training_labels : np.ndarray,
            validation_data : np.ndarray,
            validation_labels : np.ndarray,
            epochs : int = constants.DEFAULT_EPOCHS,
            learning_rate : float = constants.DEFAULT_LEARNING_RATE
    ) -> None:
        
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
            -   None.
        """

        # Controllo sulla compatibilita' di training_data e training_labels
        if (not training_data.shape[0] == training_labels.shape[0]):
            raise constants.TrainError(f"Le dimensioni del dataset [{training_data.shape[0]}] e delle labels [{training_labels.shape[0]}] di addestramento non sono compatibili.")

        # Controllo sulla compatibilita' di validation_data e validation_labels
        if (not validation_data.shape[0] == validation_labels.shape[0]):
            raise constants.TrainError(f"Le dimensioni del dataset [{validation_data.shape[0]}] e delle labels [{validation_labels.shape[0]}] per la validazione non sono compatibili.")

        training_weights = []; training_biases = []
        training_predictions = []; validation_predictions = []
        training_errors = []; validation_errors = []
        training_costs = []; validation_costs = []
        validation_weights = []; validation_biases = []

        best_net = []

        data = 0; label = 1

        start_time = time.time()
        print("Addestramento in corso...")

        for e in range(epochs):
            print(f"\nEpoca {e+1} di {epochs}")

            # TRAINING
            
            network_weights = self.weights
            network_biases = self.biases

            if self.debug:
                with np.printoptions(threshold=np.inf):
                    print("--- NETWORK WEIGHTS (start) ---\n")
                    pprint.pprint(network_weights)
                    print("\n-----")
                    print("--- NETWORK BIASES (start) ---\n")
                    pprint.pprint(network_biases)
                    print("\n-----")

            training_predictions.clear()
            training_weights.clear()
            training_biases.clear()

            for n, example in enumerate(zip(training_data, training_labels)):

                if not self.debug:
                    auxfunc.print_progress_bar(n+1, len(training_data), prefix='\tTraining:')

                # print(f"\t\tEsempio n.{n+1}")
                # print(f"\tExample: {example[data]}\n\tLabel: {example[label]}\n")

                # STEP 1: forward propagation
                training_outputs, training_activations = self.__forward_propagation(
                    example[data],
                    train=True
                )

                if self.debug:
                    with np.printoptions(threshold=np.inf):
                        print("--- NETWORK INPUTS ---\n")
                        pprint.pprint(self.inputs)
                        print("\n-----")
                        print("--- TRAINING OUTPUTS ---\n")
                        pprint.pprint(training_outputs)
                        print("\n-----")
                        print("--- TRAINING ACTIVATIONS ---\n")
                        pprint.pprint(training_activations)
                        print("\n-----")
                        print("--- TARGET ---\n")
                        pprint.pprint(example[label])
                        print("\n-----")

                training_predictions.append(training_activations[-1])

                # print("training_outputs\n")
                # pprint.pprint(training_outputs)
                # print("training_activations\n")
                # pprint.pprint(training_activations)

                # STEP 2: backpropagation per ogni esempio di training
                gw, gb = self.__back_propagation(
                    training_outputs,
                    training_activations,
                    network_weights,
                    example[label]
                )

                if self.debug:
                    with np.printoptions(threshold=np.inf):
                        print("--- BACKPROPAGATION (gradient_weights) ---\n")
                        pprint.pprint(gw)
                        print("\n-----")
                        print("--- BACKPROPAGATION (gradient_biases) ---\n")
                        pprint.pprint(gb)
                        print("\n-----")

                training_weights.append(gw)
                training_biases.append(gb)

            # end for n, example

            # STEP 3: aggiornamento dei pesi
            print("\n\tAggiornamento dei pesi in corso...")
            best_net.append(self.__update_rule(
                training_data,
                network_weights, network_biases,
                training_weights, training_biases,
                learning_rate
            ))

            self.weights = best_net[-1]["Weights"]
            self.biases = best_net[-1]["Biases"]

            # STEP 4: calcolo dell'errore per ogni esempio di training
            print("\tCalcolo dell'errore di addestramento in corso...")
            training_costs.append(self.__compute_error(np.array(training_predictions), training_labels))
            print("\tCalcolo dell'accuracy di addestramento in corso...")
            training_acc, training_percent = self.__compute_accuracy(np.array(training_predictions), training_labels)

            end_time = time.time()
            tot_time = end_time - start_time

            if (e == 0 or (e+1) % (epochs / constants.DEFAULT_EPOCHS) == 0):
                print()
                print(f"\tTempo trascorso: {tot_time:.3f} secondi")
                print(f"\tErrore di addestramento: {training_costs[-1]:.5f}")
                print(f"\tAccuracy di addestramento: {training_acc} di {len(training_labels)} ({training_percent:.2f}%)")

            print()

            # VALIDATION

            network_weights = self.weights
            network_biases = self.biases

            validation_predictions.clear()

            for n, example in enumerate(zip(validation_data, validation_labels)):

                if not self.debug:
                    auxfunc.print_progress_bar(n+1, len(validation_data), prefix='\tValidation:')

                # print(f"\t\tEsempio n.{n+1}")
                # print(f"\tExample: {example[data]}\n\tLabel: {example[label]}\n")

                # STEP 1: forward propagation
                validation_outputs, validation_activations = self.__forward_propagation(
                    example[data],
                    train=True
                )

                validation_predictions.append(validation_activations[-1])

                # STEP 2: backpropagation per ogni esempio di validation
                gw, gb = self.__back_propagation(
                    validation_outputs,
                    validation_activations,
                    network_weights,
                    example[label]
                )

                validation_weights.append(gw)
                validation_biases.append(gb)
            
            # end for n, example

            # STEP 3: aggiornamento dei pesi
            print("\n\tAggiornamento dei pesi in corso...")
            best_net.append(self.__update_rule(
                validation_data,
                network_weights, network_biases,
                validation_weights, validation_biases,
                learning_rate
            ))

            self.weights = best_net[-1]["Weights"]
            self.biases = best_net[-1]["Biases"]

            # STEP 4: calcolo errore e accuracy per ogni esempio di training
            print("\tCalcolo dell'errore di validazione in corso...")
            validation_costs.append(self.__compute_error(np.array(validation_predictions), validation_labels))
            print("\tCalcolo dell'accuracy di validazione in corso...")
            validation_acc, validation_percent = self.__compute_accuracy(np.array(validation_predictions), validation_labels)

            end_time = time.time()
            tot_time = end_time - start_time

            if (e == 0 or (e+1) % (epochs / constants.DEFAULT_EPOCHS) == 0):
                print()
                print(f"\tTempo trascorso: {tot_time:.3f} secondi")
                print(f"\tErrore di validazione: {validation_costs[-1]:.5f}")
                print(f"\tAccuracy di validazione: {validation_acc} di {len(validation_labels)} ({validation_percent:.2f}%)")

        # end for e

        print(f"L'addestramento ha impiegato {tot_time:.3f} secondi.")
        print()

        # Scelta dei parametri corrispondenti alla miglior rete (errore di validazione minimo)
        index = int(np.argmin(validation_costs, keepdims=False))
        print(index, np.min(validation_costs))
        self.weights = best_net[index]["Weights"]
        self.biases = best_net[index]["Biases"]

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

        # Utilizza la funzione softmax per ottenere valori probabilistici della predizione
        probabilities = auxfunc.softmax(prediction)

        i = 0
        print(f"Predizione della rete: {constants.ETICHETTE_CLASSI[label]}")
        for pred, prob in zip(prediction, probabilities):
            print(f'\tClasse {i}:\t{pred:.5f},\t{(prob * 100):.2f}')
            i += 1
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
# https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
# Backpropagation algorithm: https://youtu.be/sIX_9n-1UbM