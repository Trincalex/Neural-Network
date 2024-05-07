'''

    artificial_neural_network.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

'''

import constants
import auxfunc

class Neuron:
    # Attributi
    inputs = []
    weights = []
    biases = []
    scalar_product = 0
    act_fun = None

    # Metodi
    def __init__(self) -> None:
        pass

    pass

class Layer:
    # Attributi
    units = []
    size = 0

    # Metodi
    def __init__(self, s) -> None:
        self.size = s

        for i in range(s):
            self.units.append(Neuron())

        pass

    pass

class InputLayer(Layer):
    # Attributi

    # Metodi
    def __init__(self, s) -> None:
        super().__init__(s)

    pass

class HiddenLayer(Layer):
    # Attributi

    # Metodi
    def __init__(self, s) -> None:
        super().__init__(s)

    pass

class OutputLayer(Layer):
    # Attributi

    # Metodi
    def __init__(self, s) -> None:
        super().__init__(s)

    pass

class NeuralNetwork:
    # Attributi
    layers = []
    networkError = 0.0
    averageError = 0.0

    # Metodi
    def __init__(self) -> None:
        pass
    
    def train():
        pass

    def predict():
        pass

    pass