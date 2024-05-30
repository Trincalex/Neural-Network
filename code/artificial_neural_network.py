'''

    artificial_neural_network.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di una rete neurale shallow feed-forward fully-connected (aka. Multilayer Perceptron) tramite paradigma di programmazione a oggetti.
    In particolare, la classe che implementa la rete neurale (NeuralNetwork) può essere composta di uno o più strati (Layer) che, a loro volta, possono essere composti di uno o più neuroni (Neuron).

'''

# ########################################################################### #
# LIBRERIE

from artificial_layer import Layer

import constants
import auxfunc
import numpy as np
import pprint
import time
from datetime import datetime

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE NEURAL NETWORK

class NeuralNetwork:

    # ####################################################################### #
    # ATTRIBUTI DI CLASSE

    @property
    def depth(self) -> int:
        """E' la profondità della rete, cioe' il numero totale di Layer."""
        return self._depth
    # end

    @property
    def input_size(self) -> int:
        """E' la dimensione del vettore di caratteristiche di un singolo input della rete neurale."""
        return self._input_size
    # end

    @property
    def inputs(self) -> np.ndarray:
        """E' la matrice di esempi in input alla rete neurale."""
        return self._inputs
    # end

    @inputs.setter
    def inputs(self, value : np.ndarray) -> None:
        if (not isinstance(value, np.ndarray)):
            raise constants.InputLayerError("La matrice degli input deve essere di tipo 'numpy.ndarray'.")
        
        if len(value.shape) == 2:
            if (not value.shape[1] == self.input_size):
                raise constants.InputLayerError("La dimensione dei vettori delle caratteristiche degli input non e' compatibile con l'input layer.")
        elif len(value.shape) == 1:
            if (not value.size == self.input_size):
                raise constants.InputLayerError("La dimensione del vettore delle caratteristiche dell'input non e' compatibile con l'input layer.")
        else:
            raise constants.InputLayerError("La matrice degli input non e' compatibile con l'input layer.")
        
        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- NETWORK (inputs) ---\n")
                print(value.shape)
                pprint.pprint(value)
                print("\n-----\n\n")
        
        self._inputs = value
    # end

    @property
    def layers(self) -> list[Layer]:
        """E' una lista di tutti i Layer della rete."""
        return self._layers
    # end

    @property
    def weights(self) -> np.ndarray:
        """E' il vettore serializzato di tutti i pesi di tutti i neuroni della rete neurale. La sua dimensione e' pari al numero totale di pesi in ogni neurone della rete."""

        return self._weights
    # end

    @weights.setter
    def weights(self, value : np.ndarray) -> None:
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore dei pesi deve essere di tipo 'numpy.ndarray'.")
        
        # print("Neural Network:", value.size, self._weights.size)
        if (not value.size == self._weights.size):
            raise ValueError("Il vettore dei pesi non e' compatibile con questa rete neurale.")
        
        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- NETWORK (weights) ---\n")
                print(value.shape)
                pprint.pprint(value)
                print("\n-----\n\n")
        
        self._weights = value

    # end

    @property
    def biases(self) -> np.ndarray:
        """E' il vettore serializzato di tutti i bias di tutti i neuroni della rete neurale. La sua dimensione e' pari al numero totale di neuroni nei layer della rete."""

        return self._biases
    # end

    @biases.setter
    def biases(self, value : np.ndarray) -> None:
        if (not isinstance(value, np.ndarray)):
            raise ValueError("Il vettore dei bias deve essere di tipo 'numpy.ndarray'.")
        
        # print("Neural Network:", value.size, self._biases.size)
        if (not value.size == self._biases.size):
            raise ValueError("La dimensione del vettore dei bias non e' compatibile.")
        
        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- NETWORK (biases) ---\n")
                print(value.shape)
                pprint.pprint(value)
                print("\n-----\n\n")
        
        self._biases = value

    # end

    @property
    def err_fun(self) -> constants.ErrorFunctionType:
        """E' la funzione di errore utilizzata per verificare la qualità della rete neurale."""
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

    @property
    def validation_error(self):
        """..."""
        return self._validation_error
    # end

    @property
    def network_error(self):
        """..."""
        return self._network_error
    # end

    @property
    def training_accuracy(self):
        """..."""
        return self._training_accuracy
    # end

    @property
    def validation_accuracy(self):
        """..."""
        return self._validation_accuracy
    # end

    @property
    def network_accuracy(self):
        """..."""
        return self._network_accuracy
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            i_size : int,
            hidden_sizes : list[int],
            output_size : int,
            hidden_act_funs : list[constants.ActivationFunctionType] = auxfunc.sigmoid,
            output_act_fun : constants.ActivationFunctionType = auxfunc.sigmoid,
            e_fun : constants.ErrorFunctionType = auxfunc.cross_entropy_softmax,
            random_init : bool = True
    ) -> None:
        
        """
            E' il costruttore della classe NeuralNetwork.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   i_size : e' la dimensione del vettore in input alla rete neurale
            -   hidden_sizes : può essere un numero o una lista contenente la dimensione di uno o più-  hidden layer della rete neurale.
            -   output_size : e' la dimensione dell'output layer della rete neurale.
            -   hidden_act_funs : può essere una funzione o una lista contenente le funzioni di attivazione di uno o più hidden layer della rete neurale.
            -   output_act_fun : e' la funzione di attivazione dei neuroni dell'output layer.
            -   e_fun : e' la funzione di errore utilizzata per verificare la qualità della rete neurale.
            -   random_init : indica se pesi e bias della rete neurale saranno inizializzati tramite un generatore di valori casuali con seed fissato o meno.

            Returns:
            -   None
        """
        
        # Inizializzazione dell'input
        if (i_size <= 0):
            raise constants.InputLayerError("La dimensione dell'input della rete neurale deve essere maggiore di 0.")
        self._input_size = i_size
        self.inputs = np.zeros((1, self.input_size))

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
        self._layers = []
        for i in range(1, len(l_sizes)):
            # print(f'Hidden layer n.{i}')
            prev_size = l_sizes[i-1]
            actual_size = l_sizes[i]
            hl = Layer(actual_size, prev_size, l_act_funs[i], random_init)
            self.layers.append(hl)

        # Inizializzazione dell'output layer
        ol = Layer(output_size, l_sizes[-1], l_act_funs[-1], random_init)
        self.layers.append(ol)

        # Inizializzazione del vettore serializzato di pesi / bias
        weights_size = 0
        biases_size = 0

        for i in range(1, len(l_sizes)):
            weights_size += l_sizes[i-1] * l_sizes[i]
            biases_size += l_sizes[i]

        weights_size += l_sizes[-1] * output_size
        biases_size += output_size

        self._weights = np.zeros(weights_size)
        self._biases = np.zeros(biases_size)

        # Inizializzazione della profondita' della rete neurale
        # La profondita' della rete e' data dal numero di layer totali.
        self._depth = len(self.layers)

        # Inizializzazione della funzione di errore della rete
        self.err_fun = e_fun
        
        # Inizializzazione delle metriche di errore
        self._training_error = 0.0
        self._validation_error = 0.0
        self._network_error = 0.0

        # Inizializzazione delle metriche di accuracy
        self._training_accuracy = 0.0
        self._validation_accuracy = 0.0
        self._network_accuracy = 0.0

    # end
    
    # ####################################################################### #
    # METODI PRIVATI

    def __flatten_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
        
            Serializza le matrici dei pesi e dei bias di tutti i layer della rete neurale in due vettori separati della giusta dimensione.

            Returns:
            -   w : e' il vettore serializzato di tutti i pesi della rete neurale. La sua dimensione e' pari al numero totale di pesi in ogni neurone della rete.
            -   b : e' il vettore serializzato di tutti i bias della rete neurale. La sua dimensione e' pari al numero totale di neuroni nei layer della rete.
        
        """

        w = np.zeros(self.weights.shape)
        b = np.zeros(self.biases.shape)

        start_w = 0; start_b = 0
        end_w = 0; end_b = 0

        for l in self.layers:
            end_w = start_w + np.prod(l.weights.shape)
            end_b = start_b + np.prod(l.biases.shape)

            w[start_w:end_w] = l.weights.flatten()
            b[start_b:end_b] = l.biases.flatten()

            start_w = end_w
            start_b = end_b
        
        return w, b
    
    # end

    def __forward_propagation(
            self,
            x : np.ndarray,
            train : bool = False
    ) -> np.ndarray | tuple[list[np.ndarray], list[np.ndarray]]:
        
        """
            Calcola l'output complessivo della rete neurale, propagando i dati attraverso le connessioni di input dell'input layer, attraverso i calcoli intermedi degli hidden layers e, infine, attraverso l'ultimo strato dell'output layer.

            Parameters:
            -   x : il vettore di dati in input.
            -   train : serve a distinguere se l'applicazione del metodo e' per la fase di training o meno.
            
            Returns:
            -   se train=False, un numpy.ndarray contenente i valori di attivazione complessivi della rete, cioe' i valori di attivazione dell'output layer.
            -   se train=True, una prima lista contenente gli input pesati di ogni layer della rete ed una seconda lista contenente i valori di attivazione di ogni layer della rete.
        """

        # Carica input nella rete neurale.
        # Utilizza i setter di "inputs" nella classe NeuralNetwork e Layer per controllare che l'input passato sia compatibile con la rete neurale.
        self.inputs = x
        self.layers[0].inputs = x

        outputs = []
        activations = []

        for i in range(self.depth):

            if train:
                out, act = self.layers[i].activate(train=True)
                outputs.append(out)
            else:
                act = self.layers[i].activate(train=False)

            if not i == self.depth-1:
                self.layers[i+1].inputs = act
            activations.append(act)

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- NETWORK PROPAGATION (activations) ---\n")
                pprint.pprint(activations)
                print("\n-----")

        if train:
            return outputs, activations

        return activations[-1]
    
    # end

    def __delta_output_layer(
            self,
            output_layer_outputs : np.ndarray,
            output_layer_activations : np.ndarray,
            targets : np.ndarray
    ) -> np.ndarray:
        
        """
            Calcola la matrice le cui componenti sono le derivate prime parziali della funzione di costo della rete neurale rispetto agli input pesati dell'output layer.
            E' l'implementazione dell'equazione (BP1a) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.

            Parameters:
            -   output_layer_outputs : e' la matrice di input pesati dell'output layer su tutti gli esempi di training.
            -   output_layer_activations : e' la matrice di valori di attivazione dell'output layer su tutti gli esempi di training.
            -   targets : e' la matrice di tutte le etichette delle coppie del dataset.

            Returns:
            -   np.ndarray : il gradiente della funzione di costo rispetto agli input pesati dell'output layer su tutti gli esempi di training. Il numero di righe corrisponde al numero di esempi di training, mentre il numero di colonne corrisponde al numero di neuroni nell'output layer.
        """

        """
            GRADIENTE DELLA FUNZIONE DI COSTO RISPETTO ALLE ATTIVAZIONI.
            E' una matrice le cui componenti sono le singole derivate parziali della funzione di costo rispetto ai valori di attivazione dell'output layer su tutti gli esempi di training.
            Esprime quanto cambia la funzione di costo rispetto a questi valori di attivazione.
        """

        delta_Ca = self.err_fun(output_layer_activations, targets, der=True)

        """
            I pesi nell'output layer si addestrano lentamente se il valore di attivazione sono molto bassi o molto alti (per la sigmoide, ad esempio, vicino allo 0 o vicino a 1, rispettivamente).
            In questo caso, la derivata prima restituisce un valore molto vicino allo 0.
            Si dice che il neurone dell'output layer si e' saturato e, di conseguenza, il peso non si addestra piu' (o si addestra lentamente). Lo stesso vale anche per i bias della rete neurale.
            Una possibile soluzione per prevenire il rallentamento dell'apprendimento, ad esempio, potrebbe essere quella di scegliere una funzione di attivazione la cui derivata e' sempre positiva e che non si avvicina mai allo 0.
        """

        """
            DERIVATA PRIMA DEI VALORI DI ATTIVAZIONE RISPETTO AGLI INPUT PESATI.
            E' una matrice le cui componenti sono il risultato della derivata prima della funzione di attivazione dell'output layer rispetto agli input pesati dell'output layer.
        """

        delta_az = self.layers[-1].act_fun(output_layer_outputs, der=True)

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- BACKPROPAGATION (delta_output_layer) ---\n")
                pprint.pprint(delta_Ca.shape)
                pprint.pprint(delta_Ca)
                print("\n-----\n")
                pprint.pprint(delta_az.shape)
                pprint.pprint(delta_az)
                print("\n-----\n\n")
        
        # Si calcola il prodotto elemento per elemento.
        delta_Cz = np.multiply(delta_Ca, delta_az)

        return delta_Cz

    # end

    def __delta_layer(
            self,
            network_outputs : list[np.ndarray],
            network_activations : list[np.ndarray],
            delta_output_layer : np.ndarray,
            layer_index: int
    ) -> np.ndarray:
        
        """
            Calcola il vettore le cui componenti sono le derivate prime parziali della funzione di costo della rete neurale rispetto agli input pesati di un layer.
            E' l'implementazione dell'equazione (BP2) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.

            Parameters:
            -   network_outputs : la lista di output di ogni layer della rete.
            -   network_activations : la lista di valori di attivazione di ogni layer della rete.
            -   delta_output_layer : il gradiente della funzione di costo rispetto agli input pesati dell'output layer.
            -   layer_index : e' l'indice del layer scelto (corrisponde a 'l' nell'equazione proposta da Nielsen).

            Returns:
            -   np.ndarray : il gradiente della funzione di costo rispetto agli input pesati del layer scelto nella rete neurale. Il numero di righe corrisponde al numero di esempi di training, mentre il numero di colonne corrisponde al numero di neuroni nel layer scelto.
        """
        
        if layer_index == self.depth-1:
            return delta_output_layer
        
        if layer_index+1 == self.depth-1:
            delta_Cz = delta_output_layer
        else:
            delta_Cz = self.__delta_layer(
                network_outputs,
                network_activations,
                delta_output_layer,
                layer_index+1
            )

        curr_layer = self.layers[layer_index]
        next_layer = self.layers[layer_index+1]

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print(f"--- BACKPROPAGATION (delta_layer_{layer_index}) ---\n")
                pprint.pprint(delta_Cz.shape)
                pprint.pprint(delta_Cz)
                print("\n-----\n")
                pprint.pprint(next_layer.weights.shape)
                pprint.pprint(next_layer.weights)
                print("\n-----\n\n")

        delta_Ca = np.dot(delta_Cz, next_layer.weights)
        delta_az = curr_layer.act_fun(network_outputs[layer_index], der=True)

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print(f"--- BACKPROPAGATION (delta_layer_{layer_index}) ---\n")
                pprint.pprint(delta_Ca.shape)
                pprint.pprint(delta_Ca)
                print("\n-----\n")
                pprint.pprint(delta_az.shape)
                pprint.pprint(delta_az)
                print("\n-----\n\n")

        # Restituisce una matrice le cui componenti sono piccole se i corrispondenti neuroni sono vicini alla saturazione. In generale, qualsiasi input pesato di un neurone saturato si addestra lentamente (tranne nei casi in cui la matrice dei pesi può compensare questi valori piccoli).
        return np.multiply(delta_Ca, delta_az)

    # end

    def __back_propagation(
            self,
            network_outputs : list[np.ndarray],
            network_activations : list[np.ndarray],
            training_labels : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        
        """
            Aggiusta i valori dei pesi / bias della rete per diminuire il valore della funzione di costo rispetto agli esempi di training e le corrispondenti etichette in input.
            Calcola il gradiente della funzione di costo rispetto a tutti i pesi della rete utilizzando le quattro equazioni (BP1)-(BP4) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen. Tali equazioni sono una diretta conseguenza della regola della catena del calcolo multivariabile (essendo la rete fully-connected, l'aggiustamento dei pesi di un layer provoca una catena di effetti in tutti i layer successivi).
            
            Parameters:
            -   network_outputs : la lista di output di ogni layer della rete.
            -   network_activations : la lista di valori di attivazione di ogni layer della rete.
            -   training_labels : e' la matrice di tutte le etichette delle coppie del dataset.

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
            training_labels
        )

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- BACKPROPAGATION (delta_output_layer) ---\n")
                pprint.pprint(delta_output_layer.shape)
                pprint.pprint(delta_output_layer)
                print("\n-----\n\n")

        for l in range(self.depth):
            # print("layer_index:", l)

            """
                STEP 2:
                Retro-propagazione dell'errore ai layer precedenti.
            """

            delta_layer = self.__delta_layer(
                network_outputs,
                network_activations,
                delta_output_layer,
                l
            )

            if constants.DEBUG_MODE:
                with np.printoptions(threshold=np.inf):
                    print(f"--- BACKPROPAGATION (delta_layer_{l}) ---\n")
                    pprint.pprint(delta_layer.shape)
                    pprint.pprint(delta_layer)
                    print("\n-----\n\n")

            """
                STEP 3:
                Calcolo del gradiente della funzione di costo rispetto ai bias della rete neurale, cioe' come cambia la funzione di costo rispetto ai bias di tutti i neuroni della rete neurale.
                E' l'implementazione dell'equazione (BP3) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.
            """

            gl_biases = np.mean(delta_layer, axis=0)
            if constants.DEBUG_MODE:
                with np.printoptions(threshold=np.inf):
                    print(f"--- BACKPROPAGATION (layer_{l}) ---\n")
                    print("gl_biases")
                    pprint.pprint(gl_biases.shape)
                    pprint.pprint(gl_biases)
                    print("\n-----\n\n")

            gradient_biases += gl_biases.tolist()

            """
                STEP 4:
                Calcolo del gradiente della funzione di costo rispetto ai pesi della rete neurale, cioe' come cambia la funzione di costo rispetto al peso di tutte le connessioni tra due neuroni di due layer adiacenti della rete neurale.
                E' l'implementazione dell'equazione (BP4) dal Capitolo 2 del libro "Neural Networks and Deep Learning" di Michael Nielsen.
            """

            prev_layer_activations = self.inputs if l == 0 else network_activations[l-1]

            gl_weights = np.zeros((training_labels.shape[0], delta_layer.shape[1] * prev_layer_activations.shape[1]))

            for e in range(training_labels.shape[0]):
                if constants.DEBUG_MODE:
                    with np.printoptions(threshold=np.inf):
                        print(f"--- BACKPROPAGATION (example: {e}, layer: {l}) ---\n")
                        print("delta_layer")
                        pprint.pprint(delta_layer[e].shape)
                        pprint.pprint(delta_layer[e])
                        print("\n-----\n")
                        print("prev_layer_activations")
                        pprint.pprint(prev_layer_activations[e].shape)
                        pprint.pprint(prev_layer_activations[e])
                        print("\n-----\n\n")

            for e in range(training_labels.shape[0]):
                gl_weights[e] = np.outer(delta_layer[e], prev_layer_activations[e]).flatten()

            if constants.DEBUG_MODE:
                with np.printoptions(threshold=np.inf):
                    print(f"--- BACKPROPAGATION (layer: {l}) ---\n")
                    print("gl_weights")
                    pprint.pprint(gl_weights.shape)
                    pprint.pprint(gl_weights)
                    print("\n-----\n")

            gradient_weights += np.mean(gl_weights, axis=0).tolist()
        
        # end for l

        if constants.DEBUG_MODE:
            with np.printoptions(threshold=np.inf):
                print("--- GRADIENT_BIASES (mean) ---\n")
                pprint.pprint(len(gradient_biases))
                pprint.pprint(gradient_biases)
                print("\n-----\n")
                print("--- GRADIENT_WEIGHTS (mean) ---\n")
                pprint.pprint(len(gradient_weights))
                pprint.pprint(gradient_weights)
                print("\n-----\n\n")

        return np.array(gradient_weights), np.array(gradient_biases)
        
    # end

    def __update_rule(
            self,
            gradient_weights : np.ndarray,
            gradient_biases : np.ndarray,
            learning_rate : float
    ) -> dict[np.ndarray, np.ndarray]:
        
        """
            ...
            
            Parameters:
            -   ... : ...
            -   learning_rate : e' un parametro utilizzato per l'aggiornamento dei pesi che indica quanto i pesi debbano essere modificati in risposta all'errore calcolato.

            Returns:
            -   ... : ...
        """

        # Si applica la discesa del gradiente per l'aggiornamento dei pesi.
        self.weights -= learning_rate * gradient_weights

        # Si applica la discesa del gradiente per l'aggiornamento dei bias.
        self.biases -= learning_rate * gradient_biases

        # Si riporta l'aggiornamento dei pesi / bias iterativamente in tutta la rete neurale.
        start_w = 0; start_b = 0
        end_w = 0; end_b = 0

        for l in self.layers:
            end_w += np.prod(l.weights.shape)
            end_b += np.prod(l.biases.shape)

            l.weights = np.reshape(self.weights[start_w:end_w], l.weights.shape)
            l.biases = np.reshape(self.biases[start_b:end_b], l.biases.shape)

            start_w = end_w
            start_b = end_b

        return { "Weights" : self.weights, "Biases" : self.biases }

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
    ) -> dict[dict[np.ndarray, np.ndarray], float, float]:
        
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

        training_costs = []; validation_costs = []
        training_accuracies = []; validation_accuracies = []

        best_params = []

        start_time = time.time()
        print(f"\nAddestramento iniziato: {datetime.now().strftime(constants.DATE_TIME_FORMAT)}")

        # Prima di iniziare l'addestramento, recuperiamo i pesi dai layer della rete.
        self.weights, self.biases = self.__flatten_weights()

        for e in range(epochs):
            print(f"\nEpoca {e+1} di {epochs}")

            # FASE DI TRAINING
            # STEP 1: forward propagation su tutti gli esempi di addestramento
            training_outputs, training_activations = self.__forward_propagation(
                training_data,
                train=True
            )

            if constants.DEBUG_MODE:
                with np.printoptions(threshold=np.inf):
                    print("--- NETWORK TRAINING (targets) ---\n")
                    print(training_labels.shape)
                    pprint.pprint(training_labels)
                    print("\n-----\n")
                    print("--- NETWORK TRAINING (outputs) ---\n")
                    print([out.shape for out in training_outputs])
                    pprint.pprint(training_outputs)
                    print("\n-----\n")
                    print("--- NETWORK TRAINING (activations) ---\n")
                    print([act.shape for act in training_activations])
                    pprint.pprint(training_activations)
                    print("\n-----\n\n")

            # STEP 2: backpropagation su tutti gli esempi di addestramento
            gw, gb = self.__back_propagation(
                training_outputs,
                training_activations,
                training_labels
            )

            # STEP 3: aggiornamento dei pesi
            print("\n\tAggiornamento dei pesi in corso...")
            best_params.append(self.__update_rule(gw, gb, learning_rate))

            # STEP 4: calcolo dell'errore per ogni esempio di addestramento
            print("\tCalcolo dell'errore di addestramento in corso...")
            training_costs.append(
                self.__compute_error(
                    training_activations[-1],
                    training_labels
                )
            )

            t_cost_percent = training_costs[-1] / constants.NUMERO_CLASSI * 100

            # STEP 5: calcolo dell'accuracy per ogni esempio di addestramento
            print("\tCalcolo dell'accuracy di addestramento in corso...")
            t_acc, t_acc_percent = self.__compute_accuracy(
                training_activations[-1],
                training_labels
            )
            
            training_accuracies.append(t_acc)

            end_time = time.time()
            tot_time = end_time - start_time

            if (e == 0 or (e+1) % (epochs / constants.DEFAULT_EPOCHS) == 0):
                print()
                print(f"\tTempo trascorso: {tot_time:.3f} secondi")
                print(f"\tErrore di addestramento: {training_costs[-1]:.5f} ({t_cost_percent:.2f}%)")
                print(f"\tAccuracy di addestramento: {t_acc} di {len(training_labels)} ({t_acc_percent:.2f}%)")

            print()

            # FASE DI VALIDATION
            # STEP 1: forward propagation su tutti gli esempi di validazione
            validation_activations = self.__forward_propagation(validation_data)

            # STEP 2: calcolo dell'errore per ogni esempio di validazione
            print("\tCalcolo dell'errore di validazione in corso...")
            validation_costs.append(
                self.__compute_error(
                    validation_activations,
                    validation_labels
                )
            )

            v_cost_percent = validation_costs[-1] / constants.NUMERO_CLASSI * 100

            # STEP 3: calcolo dell'accuracy per ogni esempio di validazione
            print("\tCalcolo dell'accuracy di validazione in corso...")
            v_acc, v_acc_percent = self.__compute_accuracy(
                validation_activations,
                validation_labels
            )
            
            validation_accuracies.append(v_acc)

            end_time = time.time()
            tot_time = end_time - start_time

            if (e == 0 or (e+1) % (epochs / constants.DEFAULT_EPOCHS) == 0):
                print()
                print(f"\tTempo trascorso: {tot_time:.3f} secondi")
                print(f"\tErrore di validazione: {validation_costs[-1]:.5f} ({v_cost_percent:.2f}%)")
                print(f"\tAccuracy di validazione: {v_acc} di {len(validation_labels)} ({v_acc_percent:.2f}%)")

            if constants.DEBUG_MODE:
                break

        # end for e

        print(f"\nAddestramento completato: {datetime.now().strftime(constants.DATE_TIME_FORMAT)}")

        # Scelta dei parametri corrispondenti alla miglior rete (errore di validazione minimo)
        index = int(np.argmin(validation_costs, keepdims=False))

        self.weights = best_params[index]["Weights"]
        self.biases = best_params[index]["Biases"]

        self._training_error = training_costs[index]
        self._validation_error = validation_costs[index]
        t_cost_percent = self.training_error / constants.NUMERO_CLASSI * 100
        v_cost_percent = self.validation_error / constants.NUMERO_CLASSI * 100

        self._training_accuracy = training_accuracies[index]
        self._validation_accuracy = validation_accuracies[index]
        t_acc_percent = self.training_accuracy / constants.NUMERO_CLASSI * 100
        v_acc_percent = self.validation_accuracy / constants.NUMERO_CLASSI * 100

        print(f"\tTempo trascorso: {tot_time:.3f} secondi.")
        print(f"\tMiglior rete (epoca): {index+1}")
        print(f"\tMiglior rete (errore di addestramento): {self.training_error:.5f} ({t_cost_percent:.2f}%)")
        print(f"\tMiglior rete (accuracy di addestramento): {self.training_accuracy:.5f} ({t_acc_percent:.2f}%)")
        print(f"\tMiglior rete (errore di validazione): {self.validation_error:.5f} ({v_cost_percent:.2f}%)")
        print(f"\tMiglior rete (accuracy di validazione): {self.validation_accuracy:.5f} ({v_acc_percent:.2f}%)")

        return {
            "Net" : best_params[index],
            "Error" : self.validation_error,
            "Accuracy" : self._validation_accuracy
        }

    # end

    def predict(self, x : np.ndarray) -> int:
        """
            Calcola una predizione per l'input dato in base alla configurazione attuale di pesi e bias della rete neurale. Inoltre, visualizza nel terminale le probabilita' delle predizioni di tutto l'output layer utilizzando la funzione "auxfunc.softmax()".
            
            Parameters:
            -   x : il vettore di dati in input.

            Returns:
            -   label : l'indice del neurone nell'output layer che ottiene il valore di attivazione piu' alto.
        """

        labels = []
        predictions = self.__forward_propagation(x)

        for example_prediction in predictions:
            labels.append(np.argmax(example_prediction))

            # Utilizza la funzione softmax per ottenere valori probabilistici della predizione
            probabilities = auxfunc.softmax(example_prediction)

            i = 0
            print(f"Predizione della rete: {constants.ETICHETTE_CLASSI[labels[-1]]}")
            for pred, prob in zip(example_prediction, probabilities):
                print(f'\tClasse {i}:\t{pred:.5f},\t{(prob * 100):.2f}')
                i += 1
            print()

        return labels
        
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
# https://www.geeksforgeeks.org/how-to-convert-numpy-matrix-to-array/