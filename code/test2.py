'''

    test2.py
    - Alessandro Trincone
    - Mario Gabriele Carofano
    
    https://www.geeksforgeeks.org/implementation-of-neural-network-from-scratch-using-numpy/

'''

import auxfunc
import pprint
import math as mt
import random as rd
import numpy as np
import neural_network as nn
import matplotlib.pyplot as plot

# Creating data set

# A
a =[0, 0, 1, 1, 0, 0,
0, 1, 0, 0, 1, 0,
1, 1, 1, 1, 1, 1,
1, 0, 0, 0, 0, 1,
1, 0, 0, 0, 0, 1]

# B
b =[0, 1, 1, 1, 1, 0,
0, 1, 0, 0, 1, 0,
0, 1, 1, 1, 1, 0,
0, 1, 0, 0, 1, 0,
0, 1, 1, 1, 1, 0]

# C
c =[0, 1, 1, 1, 1, 0,
0, 1, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0,
0, 1, 0, 0, 0, 0,
0, 1, 1, 1, 1, 0]

# Creating labels
y =[[1, 0, 0],
[0, 1, 0],
[0, 0, 1]]

plot.imshow(np.array(a).reshape(5, 6))
plot.show()

x = [
    np.array(a).reshape(1, 30),
    np.array(b).reshape(1, 30), 
    np.array(c).reshape(1, 30)
]
 
# Labels are also converted into NumPy array
y = np.array(y)

rete = nn.Neural_Network(30, [5], 6)