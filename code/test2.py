import auxfunc
import constants
import artificial_neural_network as ann
import dataset_function as df
import numpy as np
import pprint

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

net = ann.NeuralNetwork(30, 5, 3)
# net = ann.NeuralNetwork(784, 32, 10) # una possibile configurazione per il problema della classificazione delle cifre MNIST
print("Totale neuroni:", ann.tot_neurons)

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
 
# Creating labels (one-hot)
y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# pprint.pprint(dataset)
# print(type(dataset), dataset.shape)
# print(type(dataset[0]), dataset[0].shape)
# pprint.pprint(y)
# print(type(y), y.shape)
# print(type(y[0]), y[0].shape)

# net.back_propagation(a, y, learning_rate=0.01)
prediction = net.predict(dataset[0])

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