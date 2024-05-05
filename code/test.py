import math as mt
import random as rd
import numpy as np
import neural_network as nn
import stampe as sp

# rete=nn.crea_rete(5,[3,4,2],6)
# W=rete['W']
# B=rete['B']
# d=rete['Depth']


rete = nn.Neural_Network(5,[3,4,2],6)
W = rete.get_weights()
B = rete.get_biases()
d = rete.get_depth()

for M in W:
    print(M.shape)
    sp.stampa_matrice(M)
    print()



for N in B:
    for e in N:
        print (e," ")
    print()

print(d)