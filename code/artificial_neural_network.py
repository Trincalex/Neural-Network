"""

    artificial_neural_network.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene l'implementazione di una rete neurale artificiale feed-forward fully-connected (aka. Multilayer Perceptron) tramite paradigma di programmazione a oggetti.
    In particolare, la classe che implementa la rete neurale (NeuralNetwork) può essere composta di uno o più strati (Layer).

"""

# ########################################################################### #
# LIBRERIE

from artificial_layer import Layer
from training_report import TrainingReport
from training_params import TrainingParams
import constants
import auxfunc

import numpy as np
import pprint
import time
from datetime import datetime
import copy
import matplotlib.pyplot as plot
import gc

# ########################################################################### #
# IMPLEMENTAZIONE DELLA CLASSE NEURALNETWORK

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
            raise TypeError("Il vettore dei pesi deve essere di tipo 'numpy.ndarray'.")
        
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
            raise TypeError("Il vettore dei bias deve essere di tipo 'numpy.ndarray'.")
        
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

    @property
    def training_report(self) -> TrainingReport:
        """E' un'istanza della classe TrainingReport, contenente le metriche di valutazione della fase di addestramento."""
        return self._training_report
    # end

    @property
    def training_params(self) -> TrainingParams:
        """E' un'istanza della classe TrainingParams, contenente i valori degli iper-parametri per la fase di addestramento."""
        return self._training_params
    # end

    # ####################################################################### #
    # COSTRUTTORE

    def __init__(
            self,
            i_size : int = constants.DEFAULT_INPUT_LAYER_NEURONS,
            l_sizes : list[int] = constants.DEFAULT_LAYER_NEURONS,
            l_act_funs : list[constants.ActivationFunctionType] = [auxfunc.leaky_relu, auxfunc.identity],
            e_fun : constants.ErrorFunctionType = auxfunc.cross_entropy_softmax,
            t_rep : TrainingReport = None,
            t_par : TrainingParams = None,
            random_init : bool = True
    ) -> None:
        
        """
            E' il costruttore della classe NeuralNetwork.
            Inizializza gli attributi dell'oggetto dopo la sua istanziazione.

            Parameters:
            -   i_size : e' la dimensione del vettore in input alla rete neurale
            -   l_sizes : e' una lista contenente le dimensioni di uno o piu' hidden layer e dell'output layer della rete neurale.
            -   l_act_funs : e' una lista contenente le funzioni di attivazione di uno o piu' hidden layer e dell'output layer della rete neurale.
            -   e_fun : e' la funzione di errore utilizzata per verificare la qualità della rete neurale.
            -   t_rep : e' un oggetto della classe TrainingReport che contiene alcune metriche di valutazione per la fase di addestramento della rete neurale.
            -   t_par : e' un oggetto della classe TrainingParams che contiene alcuni iper-parametri per la fase di addestramento della rete neurale.
            -   random_init : indica se i parametri (weights, biases) della rete neurale devono essere inizializzati tramite un generatore di valori casuali con seed fissato o meno.

            Returns:
            -   None.
        """
        
        # Controllo sulla dimensione dell'input layer della rete neurale.
        if not i_size > 0:
            raise constants.InputLayerError("La dimensione dell'input della rete neurale deve essere strettamente maggiore di 0.")
        
        # Inizializzazione della matrice di input della rete neurale.
        self._input_size = i_size
        self.inputs = np.zeros((1, self.input_size))

        # Controlli sulla lista delle dimensioni e delle funzioni di attivazione dei layers.
        if not isinstance(l_sizes, list):
            raise constants.LayerError("Le dimensioni dei layers devono essere passate attraverso un oggetto 'list'!")
        
        if not isinstance(l_act_funs, list):
            raise constants.LayerError("Le funzioni di attivazioni devono essere passate attraverso un oggetto 'list'!")
        
        if not len(l_sizes) == len(l_act_funs):
            raise constants.LayerError("Il numero di funzioni di attivazione deve essere uguale al numero di layer!")
        
        if (not len(l_sizes) >= 2) or (not len(l_act_funs) >= 2):
            raise constants.LayerError("I layers della rete neurale devono essere almeno 2 (e.g. un hidden layer e l'output layer)!")
        
        # Inizializzazione e aggiunta dei layers.
        self._layers = []
        for i, curr_size in enumerate(l_sizes):
            # print(f'Hidden layer n.{i}')
            prev_size = self.input_size if i == 0 else l_sizes[i-1]
            l = Layer(curr_size, prev_size, l_act_funs[i], random_init)
            self._layers.append(l)

        # Inizializzazione del vettore serializzato di pesi / bias.
        weights_size = self.input_size * l_sizes[0]
        biases_size = l_sizes[0]

        for i in range(1, len(l_sizes)):
            weights_size += l_sizes[i-1] * l_sizes[i]
            biases_size += l_sizes[i]

        self._weights = np.zeros(weights_size)
        self._biases = np.zeros(biases_size)
        self.weights, self.biases = self.__gather_weights()

        # Inizializzazione della profondita' della rete neurale.
        # La profondita' della rete e' data dal numero di layer totali.
        self._depth = len(self.layers)

        # Inizializzazione della funzione di errore della rete neurale.
        self._err_fun = e_fun
        
        # Inizializzazione delle metriche di valutazione della fase di addestramento.
        self._training_report = TrainingReport() if t_rep is None else t_rep

        # Inizializzazione degli iper-parametri per la fase di addestramento.
        self._training_params = TrainingParams() if t_par is None else t_par

    # end
    
    # ####################################################################### #
    # METODI PRIVATI

    def __gather_weights(self) -> tuple[np.ndarray, np.ndarray]:
        """
            Serializza le matrici dei pesi e dei bias di tutti i layer della rete neurale in due vettori separati della giusta dimensione. In altre parole, aggiorna le property self.weights e self.biases di questo oggetto della classe NeuralNetwork.

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

    def __scatter_weights(self) -> None:
        """
            Distribuisce i vettori dei pesi (self.weights) e dei bias (self.biases) di questo oggetto della classe NeuralNetwork nei layer della rete neurale.

            Returns:
            -   None.
        """

        start_w = 0; start_b = 0
        end_w = 0; end_b = 0

        for l in self.layers:
            end_w += np.prod(l.weights.shape)
            end_b += np.prod(l.biases.shape)

            l.weights = np.reshape(self.weights[start_w:end_w], l.weights.shape)
            l.biases = np.reshape(self.biases[start_b:end_b], l.biases.shape)

            start_w = end_w
            start_b = end_b
    
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
            training_labels : np.ndarray
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

    def __gradient_descent(
            self,
            gradient_weights : np.ndarray,
            gradient_biases : np.ndarray,
            learning_rate : float
    ) -> None:
        
        """
            Calcola l'aggiornamento dei pesi e bias della rete neurale utilizzando l'algoritmo di discesa del gradiente.
            
            Parameters:
            -   gradient_weights : e' un array contenente i gradienti dei pesi rispetto alla funzione di costo della rete neurale calcolati dalla backpropagation.
            -   gradient_biases : e' un array contenente i gradienti dei bias rispetto alla funzione di costo della rete neurale calcolati dalla backpropagation.
            -   learning_rate : e' un valore float che indica quanto i pesi debbano essere modificati in risposta all'errore calcolato.

            Returns:
            -   None.
        """

        # Si applica la discesa del gradiente per l'aggiornamento dei pesi.
        self.weights -= learning_rate * gradient_weights

        # Si applica la discesa del gradiente per l'aggiornamento dei bias.
        self.biases -= learning_rate * gradient_biases

        # Si riporta l'aggiornamento dei pesi / bias iterativamente in tutta la rete neurale.
        self.__scatter_weights()

    # end

    def __rprop_delta_layer(
        self,
        prod_gradients : np.ndarray,
        prev_delta_layer : np.ndarray,
        eta_minus : float, eta_plus : float,
        delta_min : float, delta_max : float
    ) -> np.ndarray:
        
        """
            Calcola lo 'step size' individualmente per ogni peso e bias della rete neurale.
            -   Con il termine 'step size' si indica la quantita' con cui aggiornare ogni peso / bias.
            -   In particolare, con la Rprop, gli 'step size' sono indipendenti dai valori assoluti delle derivate parziali della funzione di costo, ma dipendono solo dal loro segno.

            Parameters:
            -   prod_gradients : e' un array contenente il prodotto elemento per elemento del gradiente della funzione di costo calcolato all'epoca corrente e all'epoca precedente. Si utilizza per capire come cambia il segno.
            -   eta_minus : e' il fattore di riduzione dello step size quando il segno del gradiente cambia (tipicamente un valore inferiore a 1, es. 0.5).
            -   eta_plus : e' il fattore di incremento dello step size se il segno del gradiente rimane lo stesso (tipicamente un valore maggiore di 1, es. 1.2).
            -   delta_min : e' il valore che definisce il limite inferiore dello step size.
            -   delta_max : e' il valore che definisce il limite superiore dello step size.

            Returns:
            -   un array contenente gli step size relativi ad ogni peso / bias della rete.
        """

        return np.where(
            prod_gradients > 0,
            np.minimum(prev_delta_layer * eta_plus, delta_max),
            np.where(
                prod_gradients < 0,
                np.maximum(prev_delta_layer * eta_minus, delta_min),
                prev_delta_layer
            )
        )
    
    # end

    def __resilient_back_propagation(
            self,
            network_outputs : list[np.ndarray],
            network_activations : list[np.ndarray],
            training_labels : np.ndarray,
            prev_gw : np.ndarray, prev_gb : np.ndarray,
            prev_dlw : np.ndarray, prev_dlb : np.ndarray,
            eta_minus : float, eta_plus : float,
            delta_min : float, delta_max : float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        """
            Gli algoritmi di Resilient Backpropagation (Rprop) sono algoritmi iterativi che aggiornano i parametri di una rete neurale (weights, biases) ottimizzando la funzione di costo tramite calcolo del suo gradiente rispetto ai pesi:
            -   Risolvono il principale problema del batch learning, ovvero la necessita' di configurare a priori gli iper-parametri della rete neurale.
            -   Considerano solo il segno delle derivate parziali della funzione di costo da ottimizzare e non il loro valore assoluto. Ad ogni iterazione, ciascun peso della rete viene aumentato o diminuito se il segno della derivata parziale e' positivo o negativo, rispettivamente.
            -   Aggiustano la step size (e.g. learning rate) in base a come cambia il segno della derivata parziale della funzione di costo su iterazioni consecutive. Se il segno cambia, si diminuisce il learning rate (perche' e' stato superato il punto di minimo); altrimenti, lo si incrementa (per scendere piu' velocemente verso il punto di minimo).

            Parameters:
            -   network_outputs : la lista di output di ogni layer della rete.
            -   network_activations : la lista di valori di attivazione di ogni layer della rete.
            -   training_labels : e' la matrice di tutte le etichette delle coppie del dataset.
            -   prev_gw : e' il vettore del gradiente dei pesi dell'epoca precedente.
            -   prev_gb : e' il vettore del gradiente dei bias dell'epoca precedente.
            -   prev_dlw : e' il vettore degli step size (per i pesi) dell'epoca precedente.
            -   prev_dlb : e' il vettore degli step size (per i bias) dell'epoca precedente.
            -   eta_minus : e' il fattore di riduzione dello step size quando il segno del gradiente cambia (tipicamente un valore inferiore a 1, es. 0.5).
            -   eta_plus : e' il fattore di incremento dello step size se il segno del gradiente rimane lo stesso (tipicamente un valore maggiore di 1, es. 1.2).
            -   delta_min : e' il valore che definisce il limite inferiore dello step size.
            -   delta_max : e' il valore che definisce il limite superiore dello step size.

            Returns:
            -   gw : e' il vettore dei gradiente dei pesi dell'epoca corrente.
            -   gb : e' il vettore dei gradiente dei bias dell'epoca corrente.
            -   dl_weights : e' il vettore degli step size (per i pesi) dell'epoca corrente.
            -   dl_biases : e' il vettore degli step size (per i bias) dell'epoca corrente.
        """

        # Calcolo del vettore del gradiente dei pesi / bias.
        gw, gb = self.__back_propagation(
            network_outputs,
            network_activations,
            training_labels
        )

        # Si calcola il prodotto elemento per elemento per capire se gradiente della funzione di costo calcolato all'epoca corrente e all'epoca precedente hanno o meno lo stesso segno.
        prod_gw = np.multiply(prev_gw, gw)
        prod_gb = np.multiply(prev_gb, gb)

        # Calcolo degli step size (per i pesi).
        dl_weights = self.__rprop_delta_layer(
            prod_gw,
            prev_dlw,
            eta_minus, eta_plus,
            delta_min, delta_max
        )

        # Calcolo degli step size (per i bias).
        dl_biases = self.__rprop_delta_layer(
            prod_gb,
            prev_dlb,
            eta_minus, eta_plus,
            delta_min, delta_max
        )

        # Applicazione del weight-backtracking.
        gw          = np.where(prod_gw < 0, 0, gw)
        gb          = np.where(prod_gb < 0, 0, gb)
        u_weights   = np.where(prod_gw < 0, -np.sign(prev_gw) * prev_dlw, -np.sign(gw) * dl_weights)
        u_biases    = np.where(prod_gb < 0, -np.sign(prev_gb) * prev_dlb, -np.sign(gb) * dl_biases)

        # Si riporta l'aggiornamento dei pesi / bias iterativamente in tutta la rete neurale.
        self.weights += u_weights
        self.biases += u_biases
        self.__scatter_weights()

        return gw, gb, dl_weights, dl_biases
    
    # end
    
    # ####################################################################### #
    # METODI PUBBLICI

    def train(
            self,
            training_data : np.ndarray,
            training_labels : np.ndarray,
            validation_data : np.ndarray = None,
            validation_labels : np.ndarray = None
    ) -> list[TrainingReport]:
        
        """
            Addestra la rete neurale tramite il training set ed il validation set dati in input.
            Il processo di addestramento ripete le fasi di forward propagation, backpropagation (con calcolo della funzione di costo) e il conseguente aggiornamento dei pesi per un numero limitato di iterazioni (epochs).
            
            Parameters:
            -   training_data : una matrice numpy.ndarray contenente i dati di input per l'addestramento. Ogni riga rappresenta un esempio di addestramento.
            -   training_labels : una matrice numpy.ndarray contenente le etichette corrispondenti per i dati di addestramento. Ogni riga rappresenta l'etichetta per il rispettivo esempio di addestramento.
            -   validation_data : una matrice numpy.ndarray da utilizzare per la fase di validazione dell'addestramento. Ogni riga rappresenta un esempio di addestramento.
            -   validation_labels : una matrice numpy.ndarray da utilizzare per la fase di validazione dell'addestramento. Ogni riga rappresenta l'etichetta per il rispettivo esempio di addestramento.

            Returns:
            -   una lista di TrainingReport contenente le metriche di valutazione ottenuti durante le fasi di addestramento e validazione del modello (vedi documentazione di TrainingReport).
        """

        # Controllo sulla compatibilita' di training_data e training_labels.
        if (not training_data.shape[0] == training_labels.shape[0]):
            raise constants.TrainError(f"Le dimensioni del dataset [{training_data.shape[0]}] e delle labels [{training_labels.shape[0]}] di addestramento non sono compatibili.")

        # Controllo sulla compatibilita' di validation_data e validation_labels.
        if validation_data is not None and validation_labels is not None:
            if (not validation_data.shape[0] == validation_labels.shape[0]):
                raise constants.TrainError(f"Le dimensioni del dataset [{validation_data.shape[0]}] e delle labels [{validation_labels.shape[0]}] per la validazione non sono compatibili.")
        
        # Inizializzazione degli iper-parametri del modello.
        params = copy.deepcopy(self.training_params)
        
        # Calcolo degli indici dei mini-batch di addestramento.
        training_batches = auxfunc.compute_batches(len(training_data), params.batch_size)
        
        # Inizializzazione delle variabili necessarie all'aggiornamento dei report.
        history_report : list[TrainingReport] = []
        prev_num_epochs     = self.training_report.num_epochs
        prev_elapsed_time   = self.training_report.elapsed_time
        es_counter          = 0

        # Inizializzazione del delta_layer rispetto a pesi / bias (per la rprop)
        dlw = np.ones(self.weights.size) * params.learning_rate
        dlb = np.ones(self.biases.size) * params.learning_rate
        gw  = np.zeros(self.weights.size)
        gb  = np.zeros(self.biases.size)

        # Salvataggio del tempo di inizio dell'addestramento della rete neurale.
        start_time = time.time()
        print(f"\nAddestramento iniziato: {datetime.now().strftime(constants.PRINT_DATE_TIME_FORMAT)}")

        # Prima di iniziare l'addestramento, si recuperano tutti i pesi dai layer della rete.
        self.weights, self.biases = self.__gather_weights()

        for e in range(params.epochs):
            print(f"\nEpoca {e+1} di {params.epochs}")

            # STEP 1 : FASE DI TRAINING

            for start, end in training_batches:

                best_net_params = {
                    "Weights"   : copy.deepcopy(self.weights),
                    "Biases"    : copy.deepcopy(self.biases),
                    "Report"    : copy.deepcopy(self.training_report)
                }

                # STEP 1a: forward propagation su tutti gli esempi di addestramento
                print("\r\tEsecuzione della forward propagation...               ", end='\r')
                training_outputs, training_activations = self.__forward_propagation(
                    training_data[start:end],
                    train=True
                )

                if constants.DEBUG_MODE:
                    with np.printoptions(threshold=np.inf):
                        print("--- NETWORK TRAINING (targets) ---\n")
                        print(training_labels[start:end].shape)
                        pprint.pprint(training_labels[start:end])
                        print("\n-----\n")
                        print("--- NETWORK TRAINING (outputs) ---\n")
                        print([out.shape for out in training_outputs])
                        pprint.pprint(training_outputs)
                        print("\n-----\n")
                        print("--- NETWORK TRAINING (activations) ---\n")
                        print([act.shape for act in training_activations])
                        pprint.pprint(training_activations)
                        print("\n-----\n\n")

                # STEP 1b: aggiornamento dei parametri
                if params.rprop:

                    gw, gb, dlw, dlb = self.__resilient_back_propagation(
                        training_outputs,
                        training_activations,
                        training_labels[start:end],
                        gw, gb, dlw, dlb,
                        params.eta_minus, params.eta_plus,
                        params.delta_min, params.delta_max
                    )

                else:

                    # STEP 1b.1: backpropagation su tutti gli esempi di addestramento
                    print("\r\tEsecuzione della backpropagation...                   ", end='\r')
                    gw, gb = self.__back_propagation(
                        training_outputs,
                        training_activations,
                        training_labels[start:end]
                    )

                    print("\r\tAggiornamento dei parametri in corso...               ", end='\r')
                    self.__gradient_descent(gw, gb, params.learning_rate)

                    del gw, gb
                    gc.collect()
                
                # end if

                # STEP 1c: calcolo dell'errore di addestramento
                print("\r\tCalcolo dell'errore di addestramento in corso...      ", end='\r')
                t_cost = self.training_report.compute_error(
                    training_activations[-1],
                    training_labels[start:end],
                    self.err_fun
                )

                # STEP 1d: calcolo dell'accuracy di addestramento
                print("\r\tCalcolo dell'accuracy di addestramento in corso...    ", end='\r')
                t_acc = self.training_report.compute_accuracy(
                    training_activations[-1],
                    training_labels[start:end]
                )

                del training_outputs
                del training_activations
                gc.collect()
            
            # end for start, end
            
            # STEP 2 : FASE DI VALIDATION

            if validation_data is not None and validation_labels is not None:

                # STEP 2a: forward propagation su tutti gli esempi di validazione
                print("\r\tEsecuzione della forward propagation...               ", end='\r')
                validation_activations = self.__forward_propagation(validation_data)

                # STEP 2b: calcolo dell'errore di validazione
                print("\r\tCalcolo dell'errore di validazione in corso...        ", end='\r')
                v_cost = self.training_report.compute_error(
                    validation_activations,
                    validation_labels,
                    self.err_fun
                )

                # STEP 2c: calcolo dell'accuracy di validazione
                print("\r\tCalcolo dell'accuracy di validazione in corso...      ", end='\r')
                v_acc = self.training_report.compute_accuracy(
                    validation_activations,
                    validation_labels
                )

                del validation_activations
                gc.collect()

            else: v_cost = 0.0; v_acc = 0.0

            end_time = time.time()
            tot_time = end_time - start_time

            # STEP 3 : AGGIORNAMENTO DEL REPORT (e, se necessario, dei parametri)

            t_len = len(training_data)
            v_len = 0 if validation_data is None else len(validation_data)

            print("\r\tAggiornamento del report in corso...                  ", end='\r')
            curr_net_report = TrainingReport(
                prev_num_epochs + (e+1),
                prev_elapsed_time + tot_time,
                t_len, v_len,
                t_cost, v_cost,
                t_acc, v_acc
            )

            # STEP 4.a : verifica della qualita' dei miglioramenti (early stopping)
            v_diff = best_net_params["Report"].validation_error - curr_net_report.validation_error
            """
                Si confrontano gli errori di validazione della miglior epoca e dell'epoca corrente per capire quale configurazione di parametri (weights, biases) e' migliore. L'unica eccezione si ha per 'e == 0', cioe' la prima epoca, che deve sicuramente aggiornare il report (altrimenti non si potrebbe calcolare correttamente il minimo).
            """
            if validation_data is not None and validation_labels is not None:
                if e == 0:
                    self.training_report.update(curr_net_report)
                elif v_diff >= params.es_delta:
                    es_counter = 0
                    self.training_report.update(curr_net_report)
                else:
                    es_counter += 1
            else:
                self.training_report.update(curr_net_report)
            
            history_report.append(copy.deepcopy(curr_net_report))

            # # STEP 5 : stampa del report dell'epoca migliore
            # print("\r\t                                                      ")
            # print(repr(self.training_report))

            # # STEP 5 : stampa del report dell'ultima epoca
            # print("\r\t                                                      ")
            # print(repr(history_report[-1]))

            if constants.DEBUG_MODE:
                with np.printoptions(threshold=np.inf):
                    print("\n--- NETWORK TRAINING (validation error) ---\n")
                    print("Best:", best_net_params["Report"].validation_error)
                    print("Current:", curr_net_report.validation_error)
                    print("Diff:", v_diff)
                    print("\n-----\n")
                    print("--- NETWORK TRAINING (early stopping) ---\n")
                    print("Delta:", params.es_delta)
                    print("Patience:", params.es_patience)
                    print("Counter:", es_counter)
                    print("\n-----\n\n")

            del curr_net_report
            gc.collect()

            # STEP 4.b : verifica della qualita' dei miglioramenti (early stopping)
            if es_counter >= params.es_patience:
                break

            if constants.DEBUG_MODE:
                break

        # end for e

        print(f"\nAddestramento completato: {datetime.now().strftime(constants.PRINT_DATE_TIME_FORMAT)}")

        return history_report

    # end

    def predict(self, idTest : np.ndarray, Xtest : np.ndarray) -> list[str]:
        
        """
            Calcola le predizioni per l'input dato, in base alla configurazione attuale di pesi e bias della rete neurale.
            
            Parameters:
            -   idTest : l'array contenente gli identificativi degli esempi di testing.
            -   Xtest : la matrice di esempi di testing da elaborare.

            Returns:
            -   preds_label : la lista di etichette calcolate dalla rete neurale per ogni esempio di testing in input.
        """

        import dataset_functions as df

        # print(len(Xtest), len(idTest))
        if not len(Xtest) == len(idTest):
            raise constants.TestError("Il numero di esempi non e' compatibile con il numero di identificativi.")

        # Calcola le predizioni della rete dagli esempi di testing forniti in input.
        preds_activation = self.__forward_propagation(Xtest)

        # Calcola la distribuzione di probabilita' della predizione utilizzando la funzione softmax.
        probabilities = auxfunc.softmax(preds_activation)

        # Calcola le etichette in base alla distribuzione di probabilita' della predizione.
        preds_label = [(id, df.convert_to_label(p)) for id, p in zip(idTest, probabilities)]

        return preds_label
        
    # end

    def test(
            self,
            out_directory : str,
            idTest : np.ndarray,
            Xtest : np.ndarray,
            Ytest : np.ndarray,
            plot_mode : constants.PlotTestingMode = constants.PlotTestingMode.REPORT
    ) -> None:
        
        """
            Calcola le predizioni per l'input dato, in base alla configurazione attuale di pesi e bias della rete neurale. Quindi, confronta le etichette della ground truth con le etichette delle predizioni, mostrando i risultati in una grande tabella (vedi documentazione di plot_testing_predictions()).

            Parameters:
            -   out_directory : la directory di output dove salvare i grafici richiesti.
            -   idTest : l'array contenente gli identificativi degli esempi di testing.
            -   Xtest : la matrice contenente gli esempi di testing da elaborare. Ogni riga e' la rappresentazione dell'immagine del singolo esempio di training.
            -   Ytest : la matrice contenente le etichette corrispondenti per gli esempi di testing. Ogni riga rappresenta l'etichetta per il rispettivo esempio di testing.
            -   plot_mode : serve a distinguere per quali esempi in input e quali predizioni in output si devono disegnare i grafici (vedi documentazione di PlotTestingMode).

            Returns:
            -   None.
        """

        import plot_functions as pf

        # print(len(Xtest), len(Ytest))
        if not len(Xtest) == len(Ytest):
            raise constants.TestError("Il numero di esempi non e' compatibile con il numero di etichette.")
        
        # print(len(Xtest), len(idTest))
        if not len(Xtest) == len(idTest):
            raise constants.TestError("Il numero di esempi non e' compatibile con il numero di identificativi.")

        preds_activation = self.__forward_propagation(Xtest)
        probabilities = auxfunc.softmax(preds_activation)
        pf.plot_predictions(idTest, Xtest, Ytest, probabilities, out_directory, plot_mode)
    
    # end 

    def save_network_to_file(self, out_directory : str, out_name : str = "net.pkl" ) -> None:
        """
            Salva tutti gli iper-parametri e parametri della rete neurale in un file binario per lo storage persistente. Utilizza il modulo 'pickle' incluso in Python 3.

            Parameters:
            -   out_directory : la directory di output dove memorizzare i parametri della rete neurale.
            -   out_name : il nome del file di output.

            Returns:
            -   None.
        """

        import dill, os

        layers_sizes = []
        layers_act_funs = []

        for l in self.layers:
            layers_sizes.append(l.layer_size)
            layers_act_funs.append(l.act_fun)
        
        os.makedirs(out_directory, exist_ok=True)
    
        # Si apre il file in modalita' di scrittura per file binari.
        with open(out_directory+out_name, 'wb') as file:
            store_dict = {
                "input_size"        : self.input_size,
                "layers_sizes"      : layers_sizes,
                "layers_act_funs"   : layers_act_funs,
                "err_fun"           : self.err_fun,
                "training_report"   : self.training_report,
                "training_params"   : self.training_params,
                "weights"           : self.weights,
                "biases"            : self.biases
            }

            dill.dump(store_dict, file, dill.HIGHEST_PROTOCOL)
        # end open

        print(f"Salvataggio della rete neurale in '{out_directory+out_name}' completato.")

    # end

    # ####################################################################### #

    def __repr__(self) -> str:
        """
            Restituisce una rappresentazione dettagliata del contenuto di un oggetto della classe NeuralNetwork.
            
            Returns:
            -   una stringa contenente i dettagli dell'oggetto.
        """
        
        return f'NeuralNetwork(\n\tdepth = {self.depth},\n\tinput_size = {repr(self.input_size)},\n\tnetwork_layers = {pprint.pformat(self.layers)},\n\terr_fun = {self.err_fun},\n\ttraining_report = {pprint.pformat(self.training_report)},\n\ttraining_params = {pprint.pformat(self.training_params)}\n)'
    
    # end

    # ####################################################################### #
    # METODI STATICI

    @staticmethod
    def load_network_from_file(filename : str):
        """
            Carica la configurazione completa di iper-parametri e parametri della rete neurale direttamente da un file.

            Parameters:
            -   filename : il percorso del file dove sono memorizzati gli iper-parametri e parametri della rete neurale.

            Returns:
            -   net : la rete neurale con tutti gli iper-parametri e parametri recuperati dal file.
        """

        import dill

        with open(filename, 'rb') as file:
            store_dict = dill.load(file)
            
            input_size      = store_dict["input_size"]
            layers_sizes    = store_dict["layers_sizes"]
            layers_act_funs = store_dict["layers_act_funs"]
            err_fun         = store_dict["err_fun"]
            training_report = store_dict["training_report"]
            training_params = store_dict["training_params"]
            weights         = store_dict["weights"]
            biases          = store_dict["biases"]
        
        # end open

        net = NeuralNetwork(
            input_size,
            layers_sizes,
            layers_act_funs,
            err_fun,
            training_report,
            training_params
        )

        net.weights = copy.deepcopy(weights)
        net.biases = copy.deepcopy(biases)
        net.__scatter_weights()
    
        return net
    
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
# https://medium.com/@greyboi/serialising-all-the-functions-in-python-cd880a63b591
# https://www.kaggle.com/code/residentmario/full-batch-mini-batch-and-online-learning
# https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python
# Resilient backpropagation: https://florian.github.io/rprop/