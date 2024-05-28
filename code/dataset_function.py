import os
import numpy as np
import matplotlib.pyplot as plt
import constants

def loadDataset(training_length : int, test_length : int):
    """
        ...
        
        Parameters:
        -   ...: ...

        Returns:
        -   ...
    """

    print("Loading del dataset in corso...")

    current_dir = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset'))

    # Caricamento del file contenenti gli esempi di training
    train_file = os.path.join(dataset_dir, 'mnist_train.csv')

    # Parsing del file '.csv' e shuffling
    train_set = np.loadtxt(train_file, delimiter=','); np.random.shuffle(train_set)

    # Estrazione del numero di esempi richiesti
    train_set = train_set[:training_length]

    # Normalizzazione per valori da 0 a 1
    train_imgs = np.asfarray(train_set[:, 1:]) / constants.DIMENSIONE_PIXEL

    # Conversione delle etichette degli esempi in rappresentazione one-hot
    train_labels = convert_to_one_hot(train_set[:, 0])

    # Si ripetono gli stessi passaggi anche per il caricamento del test set
    test_file = os.path.join(dataset_dir, 'mnist_test.csv')
    test_set = np.loadtxt(test_file, delimiter=','); np.random.shuffle(test_set)
    test_set = test_set[:test_length]
    test_imgs = np.asfarray(test_set[:, 1:]) / constants.DIMENSIONE_PIXEL
    test_labels = convert_to_one_hot(test_set[:, 0])

    print("Loading del dataset completato.")
    
    return (
        train_imgs,
        train_labels,
        test_imgs,
        test_labels
    )

# end

def convert_to_one_hot(vet : np.ndarray) -> np.ndarray:
    """
        ...
        
        Parameters:
        -   ...: ...

        Returns:
        -   ...
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

def convert_to_single_label(vet : np.ndarray) -> int:
    """
        ...
        
        Parameters:
        -   ...: ...

        Returns:
        -   ...
    """

    return int(vet.argmax())

# end

def show_image(x : np.ndarray) -> None:
    """
        ...
        
        Parameters:
        -   ...: ...

        Returns:
        -   ...
    """

    xx = x.reshape((constants.DIMENSIONE_IMMAGINE, constants.DIMENSIONE_IMMAGINE))
    plt.imshow(xx, 'gray')
    plt.show()

# end

def split_dataset(
    dataset : np.ndarray,
    labels : np.ndarray,
    k : int = constants.DEFAULT_K_FOLD_VALUE
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    
    """
        Divide il dataset e le label in k sottoinsiemi.

        Parameters:
        -   dataset: il dataset da dividere.
        -   labels: le etichette del dataset da dividere.
        -   k: il numero di insiemi in cui dividere i dataset e le labels.
        
        Returns:
        -   k_fold_dataset: lista di 'k' array contenenti una porzione del dataset in input.
        -   k_fold_labels: lista di 'k' array contenenti una porzione delle labels in input.
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

# ########################################################################### #
# RIFERIMENTI

# https://numpy.org/doc/stable/reference/generated/numpy.array_split.html
# https://numpy.org/doc/stable/reference/generated/numpy.split.html
# https://zitaoshen.rbind.io/project/machine_learning/machine-learning-101-cross-vaildation/