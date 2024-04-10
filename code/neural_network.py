import random as rd
import numpy as np
import activation_function as af
import error_function as ef
    
def crea_rete(in_size,hidden_size,out_size):
    sigma=10
    weights=[]
    biases=[]
    act_fun=[]
    x=in_size                       #x: size of previus layer
    if np.isscalar(hidden_size):
        hidden_size=[hidden_size]
    for l in hidden_size:           #l: size of actual layer
        biases.append(sigma*np.random.normal(size=[l,1]))
        weights.append(sigma*np.random.normal(size=[l,x]))
        x=l
        act_fun.append(af.tanh)
    weights.append(sigma*np.random.normal(size=[out_size,x]))
    biases.append(sigma*np.random.normal(size=[out_size,1]))
    act_fun.append(af.identity)
    n_net={'W':weights,'B':biases,'ActFun':act_fun,'Depth':len(weights)}
    return n_net

def get_weights(net,i=0):
    W=net['W']
    if (i>0):
        return W[i-1]
    else:
        return W

def get_biases(net,i=0):
    B=net['B']
    if (i>0):
        return B[i-1]
    else:
        return B
    
def get_act_fun(net,i=0):
    A=net['ActFun']
    if (i>0):
        return A[i-1]
    else:
        return A

def layer_calc(w,x,b,fun,train=0):
    a = np.matmul(w,x)
    a = a+b
    z = fun(a)
    if train==0:
        return z
    else:
        return a,z

def foward_prop(net,x):
    W = get_weights(net)
    B = get_biases(net)
    AF = get_act_fun(net)
    d = net['Depth']
    z = x
    for l in range(d):
        z = layer_calc(W[l],z,B[l],AF[l])
    return z

def training_foward_prop(net,x):
    W = get_weights(net)
    B = get_biases(net)
    AF = get_act_fun(net)
    d = net['Depth']
    a=[]        #output neurone
    z=[]        #output neurone con funzione di attivazione
    d_act=[]    #derivata funzione di attivazione
    z.append(x)
    for l in range(d):
        a_loc,z_loc = layer_calc(W[l],z[l],B[l],AF[l],1)
        d_loc = AF[l](a_loc,1)
        a.append(a_loc)
        z.append(z_loc)
        d_act.append(d_loc)
    return z,d_act

def back_prop(net,x,t,err_fun):
    W = get_weights(net)
    d = net["Depth"]
    z,d_act=training_foward_prop(net,x)
    d_err=err_fun(z[-1],t,1)
    delta=[]
    delta.insert(0,d_act[-1]*d_err)
    for l in range(d-1,0,-1):
        delta_loc=d_act[l-1]*np.matmul(W[l].transpose(),delta[0])
        delta.insert(0,delta_loc)
    w_der=[]
    b_der=[]
    for l in range(d):
        w_der_loc=np.matmul(delta[l],z[l].transpose())
        w_der.append(w_der_loc)

        b_der_loc=np.sum(delta[l],1,keepdims=True)
        b_der.append(b_der_loc)
    return w_der,b_der

def gradient_calc(net,w_der,eta):
    d = net["Depth"]
    for l in range(d):
        net["W"][l] = net["W"][l]-(eta*w_der[l])

def get_accuracy_net(Z_out,T_out):
    total_cases = T_out.shape[1]
    good_cases = 0
    for i in range(total_cases):
        gold_label = np.argmax(T_out[:,i])
        net_label = np.argmax(Z_out[:,i])
        if gold_label == net_label:
            good_cases += 1
    accuracy = good_cases/total_cases
    return accuracy

def train_backpropagation(net,X_train,Y_train,X_val,Y_val,err_fun,num_epoche=0,eta=0.1):
    epoca = 0
    Z_train = foward_prop(net,X_train)
    train_err = err_fun(Z_train,Y_train)
    train_accuracy = get_accuracy_net(Z_train,Y_train)

    Z_val = foward_prop(net,X_val)
    val_err = err_fun(Z_val,Y_val)
    val_accuracy = get_accuracy_net(Z_val,Y_val)
    print("Epoca: ",-1,
          "Training Error: ",train_err,
          "Training Accuracy: ",train_accuracy,
          "Validation Error: ",val_err,
          "Validation Accuracy: ",val_accuracy)

    while epoca<num_epoche:
        w_der = back_prop(net,X_train,Y_train,err_fun)[0]

        gradient_calc(net,w_der,eta)

        Z_train = foward_prop(net,X_train)
        train_err = err_fun(Z_train,Y_train)
        train_accuracy = get_accuracy_net(Z_train,Y_train)

        Z_val = foward_prop(net,X_val)
        val_err = err_fun(Z_val,Y_val)
        val_accuracy = get_accuracy_net(Z_val,Y_val)

        if epoca == num_epoche-1:
            print("Epoca: ",epoca,
            "Training Error: ",train_err,
            "Training Accuracy: ",train_accuracy,
            "Validation Error: ",val_err,
            "Validation Accuracy: ",val_accuracy)

        epoca += 1

def rprop(net,X_train,Y_train,X_val,Y_val,err_fun,num_epoche=0,eta_minus=0.5,eta_plus=1.2,delta_zero=0.0125,delta_min=0.00001,delta_max=1):
    epoca = 0
    d = net["Depth"]
    Z_train = foward_prop(net,X_train)
    train_err = err_fun(Z_train,Y_train)
    train_accuracy = get_accuracy_net(Z_train,Y_train)

    Z_val = foward_prop(net,X_val)
    val_err = err_fun(Z_val,Y_val)
    val_accuracy = get_accuracy_net(Z_val,Y_val)
    print("Epoca: ",-1,
          "Training Error: ",train_err,
          "Training Accuracy: ",train_accuracy,
          "Validation Error: ",val_err,
          "Validation Accuracy: ",val_accuracy)
    
    der_list = []
    delta_ij = []

    for i in range(num_epoche):
        delta_ij.append([delta_zero]*d)

    while epoca < num_epoche:
        der_list = back_prop(net,X_train,Y_train,err_fun)

        for layer in range(d):
            prev_der = der_list[epoca-1][layer]
            actual_der = der_list[epoca][layer]
            der_prod = prev_der*actual_der

            delta_ij[epoca][layer] = np.where(der_prod>0, np.minimum(delta_ij[epoca-1][layer]*eta_plus, delta_max), np.where(der_prod<0, np.maximum(delta_ij[epoca-1][layer]*eta_minus, delta_min), delta_ij[epoca-1][layer]))

            net["W"][layer] = net["W"][layer] - (np.sign(der_list[epoca][layer])*delta_ij[epoca][layer])

        Z_train = foward_prop(net,X_train)
        train_err = err_fun(Z_train,Y_train)
        train_accuracy = get_accuracy_net(Z_train,Y_train)

        Z_val = foward_prop(net,X_val)
        val_err = err_fun(Z_val,Y_val)
        val_accuracy = get_accuracy_net(Z_val,Y_val)
        print("Epoca: ",epoca,
            "Training Error: ",train_err,
            "Training Accuracy: ",train_accuracy,
            "Validation Error: ",val_err,
            "Validation Accuracy: ",val_accuracy)
        
        epoca += 1

def rprop_plus(net,X_train,Y_train,X_val,Y_val,err_fun,num_epoche=0,eta_minus=0.5,eta_plus=1.2,delta_zero=0.0125,delta_min=0.00001,delta_max=1):
    epoca = 0
    d = net["Depth"]
    Z_train = foward_prop(net,X_train)
    train_err = err_fun(Z_train,Y_train)
    train_accuracy = get_accuracy_net(Z_train,Y_train)

    Z_val = foward_prop(net,X_val)
    val_err = err_fun(Z_val,Y_val)
    val_accuracy = get_accuracy_net(Z_val,Y_val)
    print("Epoca: ",-1,
          "Training Error: ",train_err,
          "Training Accuracy: ",train_accuracy,
          "Validation Error: ",val_err,
          "Validation Accuracy: ",val_accuracy)
    
    der_list = []
    delta_ij = []
    delta_wij = []

    for i in range(num_epoche):
        delta_ij.append([delta_zero]*d)
        delta_wij.append([0]*d)
    
    while epoca < num_epoche:
        der_list.append(back_prop(net,X_train,Y_train,err_fun))

        for layer in range(d):
            if epoca == 0:
                delta_wij[epoca][layer] = np.where(der_list[epoca][layer] > 0, - delta_ij[epoca][layer], np.where(der_list[epoca][layer] < 0, delta_ij[epoca][layer], 0))
                net["W"][layer] = net["W"][layer] + delta_wij[epoca][layer]

            if epoca > 0:
                prev_der = der_list[epoca-1][layer]
                actual_der = der_list[epoca][layer]
                der_prod = prev_der*actual_der

                delta_ij[epoca][layer] = np.where(der_prod>0, np.minimum(delta_ij[epoca-1][layer]*eta_plus, delta_max), np.where(der_prod<0, np.maximum(delta_ij[epoca-1][layer]*eta_minus, delta_min), delta_ij[epoca-1][layer]))
                delta_wij[epoca][layer] = np.where(der_prod >= 0, - ((np.sign(der_list[epoca][layer])) * delta_ij[epoca][layer]), - delta_wij[epoca - 1][layer])

                net["W"][layer] = net["W"][layer] + delta_wij[epoca][layer]

                der_list[epoca][layer] = np.where(der_prod<0, 0, der_list[epoca][layer])

        Z_train = foward_prop(net,X_train)
        train_err = err_fun(Z_train,Y_train)
        train_accuracy = get_accuracy_net(Z_train,Y_train)

        Z_val = foward_prop(net,X_val)
        val_err = err_fun(Z_val,Y_val)
        val_accuracy = get_accuracy_net(Z_val,Y_val)
        print("Epoca: ",epoca,
            "Training Error: ",train_err,
            "Training Accuracy: ",train_accuracy,
            "Validation Error: ",val_err,
            "Validation Accuracy: ",val_accuracy)
        
        epoca += 1