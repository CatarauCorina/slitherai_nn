import os
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn.datasets as ds
from slitherai_nn.src import nn as nn
from slitherai_nn.src import lab_nn as ln
from sklearn.model_selection import train_test_split


def dummy_data():
    class1 = [(3, 4), (4.2, 5.3), (4, 3), (6, 5), (4, 6), (3.7, 5.8),
              (3.2, 4.6), (5.2, 5.9), (5, 4), (7, 4), (3, 7), (4.3, 4.3)]
    class2 = [(-3, -4), (-2, -3.5), (-1, -6), (-3, -4.3), (-4, -5.6),
              (-3.2, -4.8), (-2.3, -4.3), (-2.7, -2.6), (-1.5, -3.6),
              (-3.6, -5.6), (-4.5, -4.6), (-3.7, -5.8)]
    labeled_data = []
    test_computation = [[(0.1, 0.2, 0.7), [1.0, 0.0, 0.0]]]

    for el in class1:
        labeled_data.append([el, [1]])
    for el in class2:
        labeled_data.append([el, [0]])

    np.random.shuffle(labeled_data)

    data, labels = zip(*labeled_data)
    data_test_comp, labels_test_comp = zip(*test_computation)
    labels_test = np.array(labels_test_comp)
    labels = np.array(labels)
    data = np.array(data, ndmin=2)
    data_test = np.array(data_test_comp, ndmin=2)
    labels_test = np.array(labels_test_comp)
    neural_network = nn.Network(bias=True, shape_in=pd.DataFrame(data_test).shape).init_network() \
        .add_layer(3, activation='relu') \
        .add_layer(3, activation='relu') \

    print(labels)
    output = neural_network.train(data_test, labels_test)
    print(output)
    # neural_network = nn.Network(data, labels, bias=True).init_network()\
    #     .add_layer(3, activation='relu')
    # print(labels)
    # output = neural_network.train()
    # print(output)


def iris_data_test():
    x_iris, y_iris = ds.load_iris(return_X_y=True)
    y_iris = np.array(y_iris, ndmin=2)
    return x_iris, y_iris


def load_mnist():
    image_size = 28  # width and height
    no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "../"
    train_data = np.loadtxt(os.path.join(data_path, "mnist_train.csv"),
                            delimiter=",")
    test_data = np.loadtxt(os.path.join(data_path, "mnist_test.csv"),
                           delimiter=",")
    fac = 0.99 / 255
    add_fac = 0.01
    train_imgs = np.asfarray(train_data[:, 1:]) * fac + add_fac
    test_imgs = np.asfarray(test_data[:, 1:]) * fac + add_fac
    train_labels = np.asfarray(train_data[:, :1])
    test_labels = np.asfarray(test_data[:, :1])

    train_labels_one_hot = one_hot(train_labels)
    train_labels_one_hot[train_labels_one_hot == 0] = 0.01
    train_labels_one_hot[train_labels_one_hot == 1] = 0.99
    test_labels_one_hot = one_hot(test_labels)
    test_labels_one_hot[test_labels_one_hot == 0] = 0.01
    test_labels_one_hot[test_labels_one_hot == 1] = 0.99
    return train_imgs, train_labels_one_hot, test_imgs, test_labels_one_hot


def load_mnist_keras():
    # Setup train and test splits
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("Training data shape: ", x_train.shape)  # (60000, 28, 28) -- 60000 images, each 28x28 pixels
    print("Test data shape", x_test.shape)  # (10000, 28, 28) -- 10000 images, each 28x28

    # Flatten the images
    image_vector_size = 28 * 28
    x_train = x_train.reshape(x_train.shape[0], image_vector_size)
    x_test = x_test.reshape(x_test.shape[0], image_vector_size)
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    return x_train, y_train, x_test, y_test


def one_hot(y):
    unique_val = np.unique(y)
    lr = np.arange(len(unique_val))
    new_y = np.array([])
    for label in y:
        one_hot = (lr == label).astype(np.int)
        if len(new_y) == 0:
            new_y = np.hstack((new_y, one_hot))
        else:
            new_y = np.vstack((new_y, one_hot))
    return new_y


def run_iris_test():
    x_iris, y_iris = ds.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x_iris, y_iris, test_size=0.1)
    targets_train = np.array(y_train, ndmin=2).T
    one_hot_train_y = one_hot(targets_train)
    targets_test = np.array(y_test, ndmin=2).T
    one_hot_test_y = one_hot(targets_test)
    neural_network_test = nn.Network(bias=True, shape_in=pd.DataFrame(x_train).shape).init_network() \
        .add_layer(8, init_type='random', activation='sigmoid')\
        .add_output(one_hot_train_y.shape, init_type='random', activation='softmax', cost_function='categorical_cross_entropy')
    neural_network_test.train_network(x_train, one_hot_train_y, x_test, one_hot_test_y, nr_epochs=300, batch_size=len(x_train))
    return


def run_mnist_test():

    mnist_x_train, mnist_y_train, mnist_x_test, mnist_y_test = load_mnist_keras()
    neural_network_mnist = nn.Network(bias=True, shape_in=pd.DataFrame(mnist_x_train).shape).init_network() \
        .add_layer(32, activation='sigmoid') \
        .add_output(mnist_y_train.shape, 'softmax', 'categorical_cross_entropy')
    neural_network_mnist.train_network(mnist_x_train,
                                       mnist_y_train,
                                       mnist_x_test, mnist_y_test,
                                       online=False,
                                       nr_epochs=1, batch_size=100)
    return


def run_compute_check():
    x = [[1, 2, 3], [1, 1, 1]]
    x_arr = np.array(x, ndmin=2)
    x_test = [[4, 5, 6], [0, 0, 0]]
    x_arr_test = np.array(x_test, ndmin=2)
    y = [[0, 1], [1, 0]]
    y_test = [[1, 0], [0, 1]]
    y_arr = np.array(y, ndmin=2)
    y_test_arr = np.array(y_test, ndmin=2)

    neural_network_debug = nn.Network(bias=True, shape_in=pd.DataFrame(x_arr).shape).init_network() \
        .add_layer(2, init_type='debug', activation='relu') \
        .add_output(y_arr.shape, activation='softmax', cost_function='categorical_cross_entropy')
    neural_network_debug.train_network(x_arr,
                                       y_arr,
                                       x_arr_test, y_test_arr,
                                       nr_epochs=10, batch_size=len(x_arr))
    return


def main():
    run_mnist_test()
    #run_iris_test()
    #run_compute_check()


if __name__ == '__main__':
    main()

