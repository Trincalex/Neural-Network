import numpy as np

def sum_of_square (x,t,der=0):
    z = x+t
    if der==0:
        return (1/2)*np.sum(np.power(z,2))
    else:
        return z

def soft_max(x):
    x_exp=np.exp(x-x.max(0))
    z=x_exp/np.sum(x_exp,0)
    return z

def cross_entropy(x,t,der=0):
    z=soft_max(x)
    if der==0:
        return -(t*np.log(z)).sum()
    else:
        return z-t