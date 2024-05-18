import os
import numpy as np
import matplotlib.pyplot as plt
import constants

def loadDataset(length : int):
    """
        ...
        
        Parameters:
        -   ...: ...

        Returns:
        -   ...
    """

    current_dir = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset'))

    train_file = os.path.join(dataset_dir, 'mnist_train.csv')
    test_file = os.path.join(dataset_dir, 'mnist_test.csv')
    
    train_set = np.loadtxt(train_file,delimiter=',', max_rows=length)
    test_set = np.loadtxt(test_file,delimiter=',', max_rows=length)
    np.random.shuffle(train_set)
    np.random.shuffle(test_set)

    # Normalizzazione per valori da 0 a 1
    train_imgs = np.asfarray(train_set[:, 1:]) / constants.DIMENSIONE_PIXEL
    test_imgs = np.asfarray(test_set[:, 1:]) / constants.DIMENSIONE_PIXEL

    train_labels = convert_to_one_hot(train_set[:, 0])
    test_labels = convert_to_one_hot(test_set[:, 0])
    
    return train_imgs.transpose(),train_labels.transpose(),test_imgs.transpose(),test_labels.transpose()

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

    xx = x.reshape((28,28))
    plt.imshow(xx,'gray')
    plt.show()

# end