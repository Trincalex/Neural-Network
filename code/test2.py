import auxfunc
import constants
from artificial_neural_network import NeuralNetwork
import dataset_function as df
import numpy as np
import pprint
import time
from datetime import datetime

# ########################################################################### #

# net = ann.NeuralNetwork(5, [30, 5, 8], 6)

# # print("Totale neuroni:", ann.tot_neurons)

# # INPUT LAYER: 5 neuroni di dimensione 1 con peso 1 e bias 0
# print("Dimensione dell'input layer:", net.input_layer.layer_size)
# for i in range(net.input_layer.layer_size):
#     n = net.input_layer.units[i]
#     print(repr(n))

# print()

# # 3 HIDDEN LAYERS
# # 1° HIDDEN LAYER: 30 neuroni di dimensione 5 con peso random e bias random
# # 2° HIDDEN LAYER: 5 neuroni di dimensione 30 con peso random e bias random
# # 3° HIDDEN LAYER: 8 neuroni di dimensione 5 con peso random e bias random
# for i in range(len(net.hidden_layers)):
#     hl = net.hidden_layers[i]
#     print(f"Dimensione dell'hidden layer n.{i+1}:", hl.layer_size)
#     for j in range(hl.layer_size):
#         n = hl.units[j]
#         print(repr(n))

# # OUTPUT LAYER: 6 neuroni di dimensione 8 con peso random e bias random
# print("Dimensione dell'output layer:", net.output_layer.layer_size)
# for i in range(net.output_layer.layer_size):
#     n = net.output_layer.units[i]
#     print(repr(n))

# ########################################################################### #

# net = NeuralNetwork(5, 3, 2, random_init=False)
# print(repr(net))

# rng = np.random.default_rng(constants.DEFAULT_RANDOM_SEED)

# Xtrain = rng.normal(loc=0.0, scale=constants.STANDARD_DEVIATION, size=(3,5))
# Ytrain = df.convert_to_one_hot(rng.integers(low=0, high=2, size=3))

# Xval = rng.normal(loc=0.0, scale=constants.STANDARD_DEVIATION, size=(3,5))
# Yval = df.convert_to_one_hot(rng.integers(low=0, high=2, size=3))

# net.train(
#      Xtrain,
#      Ytrain,
#      Xval,
#      Yval
# )

# ########################################################################### #

# net = NeuralNetwork(784, 32, 10, random_init=False)

# start_time = time.time()

# prediction = net.predict(Xtrain)

# end_time = time.time()
# tot_time = end_time - start_time
# print(f"Tempo trascorso: {tot_time:.3f} secondi")

# prediction = net.predict(dataset[1])
# prediction = net.predict(dataset[2])

# net.train(dataset, labels, dataset, labels, epochs=1)

# prediction = net.predict(dataset[0])
# prediction = net.predict(dataset[1])
# prediction = net.predict(dataset[2])

# ########################################################################### #

# for n in net.input_layer.units:
#     print(repr(n))
# print(repr(net.input_layer))
# print(repr(net))

# print(auxfunc.sigmoid(3.14, der=False))
# print(auxfunc.sigmoid(3.14, der=True))

# print(auxfunc.tanh(3.14, der=False))
# print(auxfunc.tanh(3.14, der=True))

# print(auxfunc.sum_of_squares(np.array([1,1,1]), np.array(y), der=False))
# print(auxfunc.sum_of_squares(np.array([1,1,1]), np.array(y), der=True))