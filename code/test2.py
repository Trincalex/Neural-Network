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

Xtrain, Ytrain, Xtest, Ytest = df.loadDataset(constants.COPPIE_TRAINING, constants.COPPIE_TEST)
Xtrain, Ytrain = df.split_dataset(Xtrain, Ytrain)

best_net = []

print(f"\nK-fold cross-validation iniziato: {datetime.now().strftime(constants.DATE_TIME_FORMAT)}")
start_time = time.time()

for i in range(constants.DEFAULT_K_FOLD_VALUE):

     print(f"\nFold {i+1} di {constants.DEFAULT_K_FOLD_VALUE}")

     net = NeuralNetwork(
          784, 32, 10,
          hidden_act_funs=[auxfunc.leaky_relu],
          output_act_fun=auxfunc.sigmoid,
          e_fun=auxfunc.cross_entropy_softmax
     )

     training_fold = np.concatenate([fold for j, fold in enumerate(Xtrain) if j != i])
     training_labels = np.concatenate([fold for j, fold in enumerate(Ytrain) if j != i])
     validation_fold = Xtrain[i]
     validation_labels = Ytrain[i]

     best_net.append(net.train(
          training_fold,
          training_labels,
          validation_fold,
          validation_labels
     ))

     if constants.DEBUG_MODE:
          break

# end for i

print(f"\nK-fold cross-validation completato: {datetime.now().strftime(constants.DATE_TIME_FORMAT)}")
end_time = time.time()
tot_time = end_time - start_time

# prendi la miglior rete di tutte --> vedere bene come funziona la k-fold cross validation
index = int(np.argmin([net["Error"] for net in best_net], keepdims=False))
min_error = best_net[index]["Error"]
min_error_percent = best_net[index]["Error"] / constants.NUMERO_CLASSI * 100

print(f"\tTempo trascorso: {tot_time:.3f} secondi")
print(f"\tMiglior rete (fold): {index+1}")
print(f"\tMiglior rete (errore di validazione): {min_error:.5f} ({min_error_percent:.2f}%)")

net.weights = best_net[index]["Net"]["Weights"]
net.biases = best_net[index]["Net"]["Biases"]

for test_example in zip(Xtest, Ytest):
     label = np.argmax(test_example[1])

     df.show_image(test_example[0])

     print(f"Ground truth: {constants.ETICHETTE_CLASSI[label]}")
     net.predict(test_example[0])

# ########################################################################### #

# net = NeuralNetwork(784, 32, 10, random_init=False)

# prediction = net.predict(Xtrain[0])
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