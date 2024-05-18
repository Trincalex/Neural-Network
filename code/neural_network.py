import numpy as np
import constants
import auxfunc as af

class Neural_Network:

    def __init__(self, input_size, hidden_size_list, output_size):
        weights=[]
        biases=[]
        act_fun=[]

        x = input_size                       #x: size of previus layer

        if np.isscalar(hidden_size_list):
            hidden_size_list=[hidden_size_list]

        for l in hidden_size_list:           #l: size of actual layer
            weights.append(constants.STANDARD_DEVIATION * np.random.normal(size=[l,x]))
            biases.append(constants.STANDARD_DEVIATION * np.random.normal(size=[l,1]))
            act_fun.append(af.tanh)
            x=l

        weights.append(constants.STANDARD_DEVIATION * np.random.normal(size=[output_size,x]))
        biases.append(constants.STANDARD_DEVIATION * np.random.normal(size=[output_size,1]))
        act_fun.append(af.identity)

        self.w = weights
        self.b = biases
        self.actFun = act_fun
        self.depth = len(weights)

    def set_weights(self,w):
        self.w = w

    def get_weights(self,i=0):
        if (i>0):
            return self.w[i-1]
        else:
            return self.w
        
    def set_biases(self,b):
        self.b = b
    
    def get_biases(self,i=0):
        if (i>0):
            return self.b[i-1]
        else:
            return self.b
    
    def get_actFun(self,i=0):
        if (i>0):
            return self.actFun[i-1]
        else:
            return self.actFun
        
    def get_depth(self):
        return self.depth
    
    def layer_calc(w,x,b,fun,train=0):
        a = np.matmul(w,x) + b
        z = fun(a)

        if train==0:
            return z
        else:
            return a,z
    
    def forward_prop(self,x):
        z = x

        for l in range(self.depth):
            z = self.layer_calc(
                self.w[l],
                z,
                self.b[l],
                self.actFun[l]
            )

        return z
    
    def training_forward_prop(self,x):

        z=[]        #output neurone con funzione di attivazione
        d_act=[]    #derivata funzione di attivazione

        z.append(x)

        for l in range(self.depth):
            a_loc, z_loc = self.layer_calc(
                self.w[l],
                z[l],
                self.b[l],
                self.actFun[l],
                1
            )

            d_loc = self.actFun[l](a_loc,1)

            z.append(z_loc)
            d_act.append(d_loc)

        return z,d_act
    
    def back_prop(self,x,t,err_fun):
        z,d_act=self.training_forward_prop(x)

        d_err=err_fun(z[-1],t,1)
        delta=[]
        delta.insert(0,d_act[-1]*d_err)

        for l in range(self.depth-1,0,-1):
            delta_loc=d_act[l-1]*np.matmul(self.w[l].transpose(),delta[0])
            delta.insert(0,delta_loc)

        w_der=[]
        b_der=[]

        for l in range(self.depth):
            w_der_loc=np.matmul(delta[l],z[l].transpose())
            w_der.append(w_der_loc)

            b_der_loc=np.sum(delta[l],1,keepdims=True)
            b_der.append(b_der_loc)

        return w_der,b_der
    
    def gradient_calc(self,w_der,eta):
        for l in range(self.depth):
            self.w[l] = self.w[l]-(eta*w_der[l])

    def get_accuracy_net(Z_out,T_out):
        total_cases = T_out.shape[1]
        good_cases = 0

        for i in range(total_cases):
            gold_label = np.argmax(T_out[:,i])
            net_label = np.argmax(Z_out[:,i])

            if gold_label == net_label:
                good_cases += 1

        return good_cases / total_cases
    
    def rprop(
            self,
            X_train : np.array,
            Y_train : np.array,
            X_val : np.array,
            Y_val : np.array,
            err_fun : function,
            num_epoche : int = 0,
            eta_minus : float = 0.5,
            eta_plus : float = 1.2,
            delta_zero : float = 0.0125,
            delta_min : float = 0.00001,
            delta_max : float = 1
            ) -> list:
        
        """
            Esegue la Resilient Propagation (Rprop) per l'addestramento di una rete neurale

            Parameters:
            -   X_train: Il dataset di addestramento.
            -   Y_train: I valori target corrispondenti al dataset di addestramento.
            -   X_val: Il dataset di validazione.
            -   Y_val: I valori target corrispondenti al dataset di validazione.
            -   err_fun (function): La funzione di errore utilizzata per calcolare la perdita.
            -   num_epoche: Il numero di epoche per l'addestramento. Default è 0.
            -   eta_minus: Il fattore di decremento per l'aggiornamento dei pesi. Default è 0.5.
            -   eta_plus: Il fattore di incremento per l'aggiornamento dei pesi. Default è 1.2.
            -   delta_zero: Il valore iniziale per gli incrementi dei pesi. Default è 0.0125.
            -   delta_min: Il valore minimo per gli incrementi dei pesi. Default è 0.00001.
            -   delta_max: Il valore massimo per gli incrementi dei pesi. Default è 1.

            Returns:
            -   evaluation_parameters: Una lista contenente tuple con i seguenti valori per ogni epoca:
            -   Errore di addestramento
            -   Accuratezza di addestramento
            -   Errore di validazione
            -   Accuratezza di validazione
        """

        epoca = 0
        d = self.depth
        Z_train = self.forward_prop(X_train)
        train_err = err_fun(Z_train,Y_train)
        train_accuracy = self.get_accuracy_net(Z_train,Y_train)

        Z_val = self.forward_prop(X_val)
        val_err = err_fun(Z_val,Y_val)
        val_accuracy = self.get_accuracy_net(Z_val,Y_val)
        evaluation_parameters = []
        print("Epoca: ",-1,
            "Training Error: ",train_err,
            "Training Accuracy: ",train_accuracy,
            "Validation Error: ",val_err,
            "Validation Accuracy: ",val_accuracy)
        
        evaluation_parameters.append((train_err,train_accuracy,val_err,val_accuracy))
    
        der_list = []
        delta_ij = []

        for i in range(num_epoche):
            delta_ij.append([delta_zero]*d)

        while epoca < num_epoche:
            cur_der = self.back_prop(X_train,Y_train,err_fun)
            der_list.append(cur_der)

            for layer in range(d):
                prev_der = der_list[epoca-1][layer]
                actual_der = der_list[epoca][layer]
                der_prod = prev_der*actual_der

                delta_ij[epoca][layer] = np.where(der_prod>0, np.minimum(delta_ij[epoca-1][layer]*eta_plus, delta_max), np.where(der_prod<0, np.maximum(delta_ij[epoca-1][layer]*eta_minus, delta_min), delta_ij[epoca-1][layer]))

                self.w[layer] = self.w[layer] - (np.sign(der_list[epoca][layer])*delta_ij[epoca][layer])

            Z_train = self.forward_prop(X_train)
            train_err = err_fun(Z_train,Y_train)
            train_accuracy = self.get_accuracy_net(Z_train,Y_train)

            Z_val = self.forward_prop(X_val)
            val_err = err_fun(Z_val,Y_val)
            val_accuracy = self.get_accuracy_net(Z_val,Y_val)
            print("Epoca: ",epoca,
                "Training Error: ",train_err,
                "Training Accuracy: ",train_accuracy,
                "Validation Error: ",val_err,
                "Validation Accuracy: ",val_accuracy)
        
            evaluation_parameters.append((train_err,train_accuracy,val_err,val_accuracy))
            epoca += 1
        return evaluation_parameters
    
    def k_fold_cross_validation(
            self,
            dataset,labels : np.array,
            err_fun : function,
            k : int = 10
            ) -> list:
        
        """
            Esegue la k-fold cross validation, una tecnica di validazione incrociata
            utilizzata per valutare le prestazioni di un modello predittivo.

            Parameters:
            -   dataset: Il dataset di input su cui il modello verrà addestrato e validato.
            -   labels: I valori target associati agli elementi del dataset.
            -   k: Il numero di fold in cui dividere il dataset.
                Ogni fold sarà utilizzato una volta come set di validazione
                mentre i restanti k-1 fold saranno utilizzati come set di addestramento.


            Returns:
            -   total_evaluation: Una lista contenente le medie delle metriche di valutazione
                calcolate su ciascun fold per ogni epoca.
        """

        k_fold_dataset,k_fold_labels = af.split_dataset(dataset,labels,k)
        total_evaluation = []
        for i in range(k):
            train_set = np.concatenate((k_fold_dataset[:i],k_fold_dataset[i+1:]))
            train_labels = np.concatenate((k_fold_labels[:i],k_fold_labels[i+1:]))
            validation_set = k_fold_dataset[i]
            validation_labels = k_fold_labels[i]
            evaluation_parameters = self.rprop(train_set,train_labels,validation_set,validation_labels,err_fun)
            total_evaluation.append(evaluation_parameters)

        return total_evaluation
    
    def total_average (total_evaluation : list):
        """
            Calcola la media delle metriche di validazione per ciascuna epoca.

            Parameters:
            -   total_evaluation (list of list of tuples): Una lista contenente i risultati delle metriche di validazione
                per ciascun fold della k-fold cross validation. Ogni elemento della lista esterna rappresenta un fold,
                e ogni elemento della lista interna è una tupla contenente:
                -   Errore di addestramento
                -   Accuratezza di addestramento
                -   Errore di validazione
                -   Accuratezza di validazione

            Returns:
            -   avg_train_err: Array contenente la media degli errori di addestramento per ciascuna epoca.
            -   avg_train_acc: Array contenente la media delle accuratezze di addestramento per ciascuna epoca.
            -   avg_val_err: Array contenente la media degli errori di validazione per ciascuna epoca.
            -   avg_val_acc: Array contenente la media delle accuratezze di validazione per ciascuna epoca.

        """
        num_epoche = len(total_evaluation[0])
        k = len(total_evaluation)

        avg_train_err = np.zeros(num_epoche)
        avg_train_acc = np.zeros(num_epoche)
        avg_val_err = np.zeros(num_epoche)
        avg_val_acc = np.zeros(num_epoche)

        # Somma delle metriche per ogni epoca
        for fold in total_evaluation:
            for epoch in range(num_epoche):
                train_err, train_acc, val_err, val_acc = fold[epoch]
                avg_train_err[epoch] += train_err
                avg_train_acc[epoch] += train_acc
                avg_val_err[epoch] += val_err
                avg_val_acc[epoch] += val_acc

        # Calcolo della media delle metriche
        avg_train_err /= k
        avg_train_acc /= k
        avg_val_err /= k
        avg_val_acc /= k

        # Restituzione delle medie delle metriche per ciascuna epoca
        return avg_train_err, avg_train_acc, avg_val_err, avg_val_acc


    def train():
        pass

    def predict():
        pass
    
    
# def crea_rete(input_size,hidden_size,output_size):
#     sigma=10
#     weights=[]
#     biases=[]
#     act_fun=[]
#     x=input_size                       #x: size of previus layer
#     if np.isscalar(hidden_size):
#         hidden_size=[hidden_size]
#     for l in hidden_size:           #l: size of actual layer
#         biases.append(sigma*np.random.normal(size=[l,1]))
#         weights.append(sigma*np.random.normal(size=[l,x]))
#         x=l
#         act_fun.append(auxfunc.tanh)
#     weights.append(sigma*np.random.normal(size=[output_size,x]))
#     biases.append(sigma*np.random.normal(size=[output_size,1]))
#     act_fun.append(auxfunc.identity)
#     n_net={'W':weights,'B':biases,'ActFun':act_fun,'Depth':len(weights)}
#     return n_net

# def get_weights(net,i=0):
#     W=net['W']
#     if (i>0):
#         return W[i-1]
#     else:
#         return W

# def get_biases(net,i=0):
#     B=net['B']
#     if (i>0):
#         return B[i-1]
#     else:
#         return B
    
# def get_act_fun(net,i=0):
#     A=net['ActFun']
#     if (i>0):
#         return A[i-1]
#     else:
#         return A

# def layer_calc(w,x,b,fun,train=0):
#     a = np.matmul(w,x)
#     a = a+b
#     z = fun(a)
#     if train==0:
#         return z
#     else:
#         return a,z

# def forward_prop(net,x):
#     W = get_weights(net)
#     B = get_biases(net)
#     AF = get_act_fun(net)
#     d = net['Depth']
#     z = x
#     for l in range(d):
#         z = layer_calc(W[l],z,B[l],AF[l])
#     return z

# def training_forward_prop(net,x):
#     W = get_weights(net)
#     B = get_biases(net)
#     AF = get_act_fun(net)
#     d = net['Depth']
#     #   a=[]        #output neurone
#     z=[]        #output neurone con funzione di attivazione
#     d_act=[]    #derivata funzione di attivazione
#     z.append(x)
#     for l in range(d):
#         a_loc,z_loc = layer_calc(W[l],z[l],B[l],AF[l],1)
#         d_loc = AF[l](a_loc,1)
#         #   a.append(a_loc)
#         z.append(z_loc)
#         d_act.append(d_loc)
#     return z,d_act

# def back_prop(net,x,t,err_fun):
#     W = get_weights(net)
#     d = net["Depth"]
#     z,d_act=training_forward_prop(net,x)
#     d_err=err_fun(z[-1],t,1)
#     delta=[]
#     delta.insert(0,d_act[-1]*d_err)
#     for l in range(d-1,0,-1):
#         delta_loc=d_act[l-1]*np.matmul(W[l].transpose(),delta[0])
#         delta.insert(0,delta_loc)
#     w_der=[]
#     b_der=[]
#     for l in range(d):
#         w_der_loc=np.matmul(delta[l],z[l].transpose())
#         w_der.append(w_der_loc)

#         b_der_loc=np.sum(delta[l],1,keepdims=True)
#         b_der.append(b_der_loc)
#     return w_der,b_der

# def gradient_calc(net,w_der,eta):
#     d = net["Depth"]
#     for l in range(d):
#         net["W"][l] = net["W"][l]-(eta*w_der[l])

# def get_accuracy_net(Z_out,T_out):
#     total_cases = T_out.shape[1]
#     good_cases = 0
#     for i in range(total_cases):
#         gold_label = np.argmax(T_out[:,i])
#         net_label = np.argmax(Z_out[:,i])
#         if gold_label == net_label:
#             good_cases += 1
#     accuracy = good_cases/total_cases
#     return accuracy

# def train_backpropagation(net,X_train,Y_train,X_val,Y_val,err_fun,num_epoche=0,eta=0.1):
#     epoca = 0
#     Z_train = forward_prop(net,X_train)
#     train_err = err_fun(Z_train,Y_train)
#     train_accuracy = get_accuracy_net(Z_train,Y_train)

#     Z_val = forward_prop(net,X_val)
#     val_err = err_fun(Z_val,Y_val)
#     val_accuracy = get_accuracy_net(Z_val,Y_val)
#     print("Epoca: ",-1,
#           "Training Error: ",train_err,
#           "Training Accuracy: ",train_accuracy,
#           "Validation Error: ",val_err,
#           "Validation Accuracy: ",val_accuracy)

#     while epoca<num_epoche:
#         w_der = back_prop(net,X_train,Y_train,err_fun)[0]

#         gradient_calc(net,w_der,eta)

#         Z_train = forward_prop(net,X_train)
#         train_err = err_fun(Z_train,Y_train)
#         train_accuracy = get_accuracy_net(Z_train,Y_train)

#         Z_val = forward_prop(net,X_val)
#         val_err = err_fun(Z_val,Y_val)
#         val_accuracy = get_accuracy_net(Z_val,Y_val)

#         if epoca == num_epoche-1:
#             print("Epoca: ",epoca,
#             "Training Error: ",train_err,
#             "Training Accuracy: ",train_accuracy,
#             "Validation Error: ",val_err,
#             "Validation Accuracy: ",val_accuracy)

#         epoca += 1

# def rprop(net,X_train,Y_train,X_val,Y_val,err_fun,num_epoche=0,eta_minus=0.5,eta_plus=1.2,delta_zero=0.0125,delta_min=0.00001,delta_max=1):
#     epoca = 0
#     d = net["Depth"]
#     Z_train = forward_prop(net,X_train)
#     train_err = err_fun(Z_train,Y_train)
#     train_accuracy = get_accuracy_net(Z_train,Y_train)

#     Z_val = forward_prop(net,X_val)
#     val_err = err_fun(Z_val,Y_val)
#     val_accuracy = get_accuracy_net(Z_val,Y_val)
#     print("Epoca: ",-1,
#           "Training Error: ",train_err,
#           "Training Accuracy: ",train_accuracy,
#           "Validation Error: ",val_err,
#           "Validation Accuracy: ",val_accuracy)
    
#     der_list = []
#     delta_ij = []

#     for i in range(num_epoche):
#         delta_ij.append([delta_zero]*d)

#     while epoca < num_epoche:
#         cur_der = back_prop(net,X_train,Y_train,err_fun)
#         der_list.append(cur_der)

#         for layer in range(d):
#             prev_der = der_list[epoca-1][layer]
#             actual_der = der_list[epoca][layer]
#             der_prod = prev_der*actual_der

#             delta_ij[epoca][layer] = np.where(der_prod>0, np.minimum(delta_ij[epoca-1][layer]*eta_plus, delta_max), np.where(der_prod<0, np.maximum(delta_ij[epoca-1][layer]*eta_minus, delta_min), delta_ij[epoca-1][layer]))

#             net["W"][layer] = net["W"][layer] - (np.sign(der_list[epoca][layer])*delta_ij[epoca][layer])

#         Z_train = forward_prop(net,X_train)
#         train_err = err_fun(Z_train,Y_train)
#         train_accuracy = get_accuracy_net(Z_train,Y_train)

#         Z_val = forward_prop(net,X_val)
#         val_err = err_fun(Z_val,Y_val)
#         val_accuracy = get_accuracy_net(Z_val,Y_val)
#         print("Epoca: ",epoca,
#             "Training Error: ",train_err,
#             "Training Accuracy: ",train_accuracy,
#             "Validation Error: ",val_err,
#             "Validation Accuracy: ",val_accuracy)
        
#         epoca += 1

# def rprop_plus(net,X_train,Y_train,X_val,Y_val,err_fun,num_epoche=0,eta_minus=0.5,eta_plus=1.2,delta_zero=0.0125,delta_min=0.00001,delta_max=1):
#     epoca = 0
#     d = net["Depth"]
#     Z_train = forward_prop(net,X_train)
#     train_err = err_fun(Z_train,Y_train)
#     train_accuracy = get_accuracy_net(Z_train,Y_train)

#     Z_val = forward_prop(net,X_val)
#     val_err = err_fun(Z_val,Y_val)
#     val_accuracy = get_accuracy_net(Z_val,Y_val)
#     print("Epoca: ",-1,
#           "Training Error: ",train_err,
#           "Training Accuracy: ",train_accuracy,
#           "Validation Error: ",val_err,
#           "Validation Accuracy: ",val_accuracy)
    
#     der_list = []
#     delta_ij = []
#     delta_wij = []

#     for i in range(num_epoche):
#         delta_ij.append([delta_zero]*d)
#         delta_wij.append([0]*d)
    
#     while epoca < num_epoche:
#         der_list.append(back_prop(net,X_train,Y_train,err_fun))

#         for layer in range(d):
#             if epoca == 0:
#                 delta_wij[epoca][layer] = np.where(der_list[epoca][layer] > 0, - delta_ij[epoca][layer], np.where(der_list[epoca][layer] < 0, delta_ij[epoca][layer], 0))
#                 net["W"][layer] = net["W"][layer] + delta_wij[epoca][layer]

#             if epoca > 0:
#                 prev_der = der_list[epoca-1][layer]
#                 actual_der = der_list[epoca][layer]
#                 der_prod = prev_der*actual_der

#                 delta_ij[epoca][layer] = np.where(der_prod>0, np.minimum(delta_ij[epoca-1][layer]*eta_plus, delta_max), np.where(der_prod<0, np.maximum(delta_ij[epoca-1][layer]*eta_minus, delta_min), delta_ij[epoca-1][layer]))
#                 delta_wij[epoca][layer] = np.where(der_prod >= 0, - ((np.sign(der_list[epoca][layer])) * delta_ij[epoca][layer]), - delta_wij[epoca - 1][layer])

#                 net["W"][layer] = net["W"][layer] + delta_wij[epoca][layer]

#                 der_list[epoca][layer] = np.where(der_prod<0, 0, der_list[epoca][layer])

#         Z_train = forward_prop(net,X_train)
#         train_err = err_fun(Z_train,Y_train)
#         train_accuracy = get_accuracy_net(Z_train,Y_train)

#         Z_val = forward_prop(net,X_val)
#         val_err = err_fun(Z_val,Y_val)
#         val_accuracy = get_accuracy_net(Z_val,Y_val)
#         print("Epoca: ",epoca,
#             "Training Error: ",train_err,
#             "Training Accuracy: ",train_accuracy,
#             "Validation Error: ",val_err,
#             "Validation Accuracy: ",val_accuracy)
        
#         epoca += 1