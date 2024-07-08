"""

    dataset_functions.py
    - Alessandro Trincone
    - Mario Gabriele Carofano

    Questo file contiene tutte le funzioni per caricare e gestire il dataset.

"""

# ########################################################################### #
# LIBRERIE

import os
import numpy as np
import matplotlib.pyplot as plt
import constants

# ########################################################################### #
# DEFINIZIONE DELLE FUNZIONI

def loadDataset(
        training_length : int,
        test_length : int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    """
        E' la funzione per caricare il dataset mnist.
        
        Parameters:
        -   training_length : intero rappresentante la grandezza del training set.
        -   test_length : intero rappresentante la grandezza del test set.

        Returns:
        -   train_ids : array che contiene gli id degli elementi del training set.
        -   train_imgs : array che contiene gli elementi del training set.
        -   train_labels : array che contiene le etichette degli elementi del training set.
        -   test_ids : array che contiene gli elementi del test set.
        -   test_imgs : array che contiene gli elementi del test set.
        -   test_labels : array che contiene le etichette degli elementi del test set.
    """

    print("Loading del dataset in corso...                                   ", end='\r')

    current_dir = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset'))

    # Caricamento del file contenenti gli esempi di training.
    train_file = os.path.join(dataset_dir, 'mnist_train.csv')

    # Parsing del file '.csv' e aggiunta di un valore identificativo del singolo esempio.
    ts = np.loadtxt(train_file, delimiter=',')
    ids = np.array(range(ts.shape[0]))
    # print(ts[0], type(ts[0]))
    # print(ts.shape, ts[0].shape)

    train_set = np.concatenate((ids[:, None], ts), axis=1)

    # Shuffling degli elementi del training set.
    np.random.shuffle(train_set)

    # Estrazione del numero di esempi richiesti.
    train_set = train_set[:training_length]

    # Normalizzazione per valori da 0 a 1
    train_imgs = np.asfarray(train_set[:, 2:]) / constants.PIXEL_INTENSITY_LEVELS

    # Conversione delle etichette degli esempi in rappresentazione one-hot.
    train_labels = convert_to_one_hot(train_set[:, 1])

    # Recupero degli identificativi delle cifre del training set.
    train_ids = train_set[:, 0].astype(int)

    # Si ripetono gli stessi passaggi anche per il caricamento del test set.
    test_file = os.path.join(dataset_dir, 'mnist_test.csv')

    ts = np.loadtxt(test_file, delimiter=',')
    ids = np.array(range(ts.shape[0]))
    test_set = np.concatenate((ids[:, None], ts), axis=1)
    np.random.shuffle(test_set)

    test_set = test_set[:test_length]
    test_imgs = np.asfarray(test_set[:, 2:]) / constants.PIXEL_INTENSITY_LEVELS
    test_labels = convert_to_one_hot(test_set[:, 1])
    test_ids = test_set[:, 0].astype(int)

    # print("id:", test_set[0,0], "label:", test_set[0,1])
    # print("label:", test_labels[0])
    # print("id:", test_ids[0])

    print("\r\nLoading del dataset completato.                               ")
    
    return (
        train_ids,
        train_imgs,
        train_labels,
        test_ids,
        test_imgs,
        test_labels
    )

# end

def split_dataset(
    dataset : np.ndarray,
    labels : np.ndarray,
    k : int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    
    """
        Divide il dataset e le label in k sottoinsiemi.

        Parameters:
        -   dataset: il dataset da dividere.
        -   labels: le etichette del dataset da dividere.
        -   k: il numero di insiemi in cui dividere i dataset e le labels.
        
        Returns:
        -   k_fold_dataset: lista di 'k' array contenenti una porzione degli esempi del dataset in input.
        -   k_fold_labels: lista di 'k' array contenenti una porzione delle labels corrispondenti agli esempi.
    """

    # print(type(dataset), dataset, dataset.shape)
    # print("\n-----\n")
    # print(type(labels), labels, labels.shape)

    """
        Il metodo 'array_split' divide l'array in input in 'k' sotto-array, anche se 'k' non divide equamente l'asse scelto di dimensione 'l'. In questo caso, si hanno 'l % k' sotto-array di dimensione 'l//k + 1' e i restanti sotto-array di dimensione 'l//k'.
        Il parametro 'axis' indica su quale dimensione della matrice effettuare il partizionamento (per axis=0, si intende la dimensione delle righe).
    """
    k_fold_dataset = np.array_split(dataset, k, axis=0)
    k_fold_labels = np.array_split(labels, k, axis=0)

    # print(type(k_fold_dataset[0]), k_fold_dataset[0], k_fold_dataset[0].shape)
    # print("\n-----\n")
    # print(type(k_fold_labels[0]), k_fold_labels[0], k_fold_labels[0].shape)

    return k_fold_dataset, k_fold_labels

# end

def convert_to_one_hot(vet : np.ndarray) -> np.ndarray:
    """
        Converte un vettore di interi (un'etichetta per ogni esempio del dataset) nella rappresentazione one-hot in forma di matrice (un vettore one-hot per ogni esempio di training).
        
        Parameters:
        -   vet : il vettore di interi da convertire nella rappresentazione one-hot.

        Returns:
        -   one_hot_matrix : una matrice contenente la rappresentazione one-hot dell'etichetta di ogni esempio di training.
    """
    
    num_label = len(vet)

    one_hot_matrix = np.zeros((num_label, constants.NUMERO_CLASSI), dtype=int)

    for i in range(num_label):
        curr_class = int(vet[i])
        curr_one_hot = np.zeros(constants.NUMERO_CLASSI)
        curr_one_hot[curr_class] = 1
        one_hot_matrix[i] = curr_one_hot

    one_hot_matrix = one_hot_matrix.astype(int)

    return one_hot_matrix

# end

def convert_to_label(vet : np.ndarray) -> str:
    """
        Converte la distribuzione di probabilita' dell'output della rete neurale nell'etichetta corrispondente alla predizione.
        
        Parameters:
        -   vet : la distribuzione di probabilita' dell'output della rete neurale.

        Returns:
        -   l'etichetta corrispondente alla predizione della rete neurale.
    """

    return constants.ETICHETTE_CLASSI[np.argmax(vet)]

# end

# ########################################################################### #
# RIFERIMENTI

# https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
# https://numpy.org/doc/stable/reference/generated/numpy.split.html
# https://zitaoshen.rbind.io/project/machine_learning/machine-learning-101-cross-vaildation/