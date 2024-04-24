import math as mt
import random as rd
import numpy as np
import neural_network as nn
import stampe as sp

rete=nn.crea_rete(5,[3,4,2],6)

W=rete['W']
for M in W:
    print(M.shape)
    sp.stampa_matrice(M)
    print()

B=rete['B']

for N in B:
    for e in N:
        print (e," ")
    print()

print(rete['Depth'])