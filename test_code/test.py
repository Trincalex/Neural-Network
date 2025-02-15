import auxfunc
import pprint
import math as mt
import random as rd
import numpy as np
import neural_network as nn
import matplotlib.pyplot as plot
import auxfunc as af
import dataset_functions as ds
import pprint
import constants

# rete=nn.crea_rete(5,[3,4,2],6)
# W=rete['W']
# B=rete['B']
# d=rete['Depth']

# data = np.arange(1,101)
# test = np.arange(1,101)
# k_data,k_test = af.split_dataset(data,test,10)
# print(k_data[1])
# print(k_test[1])
# senza = np.concatenate((k_data[:1],k_data[1+1:]))
# print(senza)

Xtrain,Ytrain, Xtest,Ytest = ds.loadDataset(constants.COPPIE_TRAINING)

for i in range(10):
    print(Xtrain.shape)
    print(Ytrain.shape)
    index = int(i * np.random.normal())
    # print(Ytrain[:, index])
    print(ds.convert_to_label(Ytrain[:, index]))
    x=Xtrain[:,index]
    ix=np.reshape(x,(constants.DIMENSIONE_IMMAGINE, constants.DIMENSIONE_IMMAGINE))
    ds.show_image(ix)

#y = Ytrain[10000]
#print(y)

# rete = nn.Neural_Network(5,[3,4,2],6)

# for i in range(len(rete.get_weights())):
#     print("Shape del %do vettore dei pesi: " % (i+1), rete.get_weights()[i].shape)

# print("\n")
# pprint.pprint(rete.get_weights())
# pprint.pprint(rete.get_weights())

# print("\n")
# print(rete.get_depth())

# ########################################################################### #
# RIFERIMENTI

# https://www.geeksforgeeks.org/python-output-formatting/
# https://numpy.org/doc/1.25/reference/random/generated/numpy.random.shuffle.html