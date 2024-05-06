import auxfunc
import pprint
import math as mt
import random as rd
import numpy as np
import neural_network as nn
import matplotlib.pyplot as plot

# rete=nn.crea_rete(5,[3,4,2],6)
# W=rete['W']
# B=rete['B']
# d=rete['Depth']

rete = nn.Neural_Network(5,[3,4,2],6)

for i in range(len(rete.get_weights())):
    print("Shape del %do vettore dei pesi: " % (i+1), rete.get_weights()[i].shape)

print("\n")
pprint.pprint(rete.get_weights())
pprint.pprint(rete.get_weights())

print("\n")
print(rete.get_depth())

'''
Riferimenti
-   https://www.geeksforgeeks.org/python-output-formatting/

'''