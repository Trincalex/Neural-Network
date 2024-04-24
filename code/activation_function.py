import numpy as np

def sigmoid(x,der=0):
    y=1/(1+np.exp(-x))
    if der==0:
        return y
    else:
        return y*(1-y)

def tanh(x,der=0):
    a=np.exp(2*x)
    y=(a-1)/(a+1)
    if der==0:
        return y
    else:
        return 1-y*y
    
def identity(x,der=0):
    if der==0:
        return x
    else:
        return 1