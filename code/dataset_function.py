import os
import numpy as np
import matplotlib.pyplot as plt

def loadDataset():
    current_dir = os.path.dirname(__file__)
    dataset_dir = os.path.abspath(os.path.join(current_dir, '..', 'dataset'))

    train_file = os.path.join(dataset_dir, 'mnist_train.csv')
    test_file = os.path.join(dataset_dir, 'mnist_test.csv')
    
    train_set = np.loadtxt(train_file,delimiter=',')
    test_set = np.loadtxt(test_file,delimiter=',')

    train_imgs = np.asfarray(train_set[:, 1:]) / 255
    test_imgs = np.asfarray(test_set[:, 1:]) / 255

    train_labels = convert_to_one_hot(train_set[:, 0])
    test_labels = convert_to_one_hot(test_set[:, 0])
    
    return train_imgs.transpose(),train_labels.transpose(),test_imgs.transpose(),test_labels.transpose()

def convert_to_one_hot(vet):
    num_label = len(vet)
    num_class = 10

    one_hot_matrix = np.zeros((num_label,num_class),dtype=int)

    for i in range(0, num_label):   
        curr_class = int(vet[i])
        curr_one_hot = np.zeros(10)
        curr_one_hot[curr_class] = 1
        one_hot_matrix[i] = curr_one_hot

    one_hot_matrix = one_hot_matrix.astype(int)
    return one_hot_matrix

def show_image(x):
    xx=x.reshape((28,28))
    plt.imshow(xx,'gray')
    plt.show()