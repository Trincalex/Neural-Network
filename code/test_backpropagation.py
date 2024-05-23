import auxfunc
import constants
from artificial_neural_network import NeuralNetwork
import dataset_function as df
import numpy as np
import pprint

net = NeuralNetwork(30, 5, 3)

# # La matrice dei pesi dell'hidden layer (l=0) di dimensione (5, 30).
# l = 0
# print(net.layers[l].weights)

# # Il peso della connessione tra il primo neurone (j=0) dell'hidden layer (l=0) e il primo neurone (k=0) dell'input layer.
# l = 0; j = 0; k = 0
# print(net.layers[l].weights[j][k])

# # Il vettore colonna dei bias del livello 0 (hidden layer) di dimensione (5, 1).
# l = 0
# print(net.layers[l].biases)

# # Il bias del primo neurone (j=0) dell'hidden layer (l=0).
# l = 0; j = 0
# print(net.layers[l].biases[j])

# ########################################################################### #

# Creating dataset
# Letter A
a = [0, 0, 1, 1, 0, 0,
     0, 1, 0, 0, 1, 0,
     1, 1, 1, 1, 1, 1,
     1, 0, 0, 0, 0, 1,
     1, 0, 0, 0, 0, 1]

# Letter B
b = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 1, 0,
     0, 1, 1, 1, 1, 0]

# Letter C
c = [0, 1, 1, 1, 1, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 0, 0, 0, 0,
     0, 1, 1, 1, 1, 0]

dataset = np.array([a, b, c])
labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# pprint.pprint(dataset)
# print(type(dataset), dataset.shape)
# print(type(dataset[0]), dataset[0].shape)
# pprint.pprint(y)
# print(type(y), y.shape)
# print(type(y[0]), y[0].shape)

rng = np.random.default_rng(0)

hidden_weights = np.reshape(rng.normal(loc=0.0, scale=1.0, size=150), (5, 30))
hidden_biases = np.reshape(rng.normal(loc=0.0, scale=1.0, size=5), (5, 1))
output_weights = np.reshape(rng.normal(loc=0.0, scale=1.0, size=15), (3, 5))
output_biases = np.reshape(rng.normal(loc=0.0, scale=1.0, size=3), (3, 1))

input_activations = a

print("--- NEURAL NETWORK ---\n")

# print(net.weights, "\n")
# print(net.biases, "\n")

network_outputs, network_activations = net.forward_propagation(input_activations, train=True)

# print(net.inputs, "\n")
# print(network_outputs, "\n")
# print(network_activations, "\n")

w1, b1 = net.back_propagation0(network_outputs, network_activations, labels[0])

print("-----\n")

print("--- BACKPROPAGATION TEST ---\n")

w2, b2 = net.back_propagation(labels[0])

min_length = min(len(w1), len(w2))
# Conta gli elementi uguali posizione per posizione
count = sum(1 for i in range(min_length) if w1[i] == w2[i])
print(f"Number of matching weights: {count} out of {len(w1)}")

min_length = min(len(b1), len(b2))
# Conta gli elementi uguali posizione per posizione
count = sum(1 for i in range(min_length) if b1[i] == b2[i])
print(f"Number of matching biases: {count} out of {len(b1)}")

# print(hidden_weights, "\n")
# print(output_weights, "\n")
# print(hidden_biases, "\n")
# print(output_biases, "\n")

hidden_outputs = np.reshape(np.dot(hidden_weights, input_activations), (-1,1)) + hidden_biases
hidden_activations = np.array([auxfunc.sigmoid(value) for value in hidden_outputs])
final_outputs = np.dot(output_weights, hidden_activations) + output_biases
final_activations = np.array([auxfunc.sigmoid(value) for value in final_outputs])

# print(input_activations, "\n")
# print(hidden_outputs, "\n")
# print(final_outputs, "\n")
# print(hidden_activations, "\n")
# print(final_activations, "\n")

print("-----\n")

# ########################################################################### #
# RIFERIMENTI

# https://builtin.com/data-science/numpy-random-seed