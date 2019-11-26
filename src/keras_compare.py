import os
import numpy as np
import tensorflow as tf
import data_preproc as dp
from tensorflow import keras as ks


from sklearn import datasets as ds
from sklearn.model_selection import train_test_split


def construct_net(input_shape, output_shape):
    model = ks.Sequential()

    # The input layer requires the special input_shape parameter which should match
    # the shape of our training data.
    model.add(ks.layers.Dense(32, input_dim=input_shape, activation='sigmoid'))
    model.add(ks.layers.Dense(output_shape, activation='softmax'))

    #
    # model.add(ks.layers.Dense(units=100, activation='relu', input_shape=(input_shape,)))
    # model.add(ks.layers.Dense(units=output_shape, activation='softmax'))
    return model


def construct_net_svhn(input_shape, output_shape):
    model = ks.Sequential()

    model.add(ks.layers.Dense(200, input_dim=input_shape, activation='relu'))
    model.add(ks.layers.Dropout(rate=0.4))
    model.add(ks.layers.Dense(200, activation='relu'))

    model.add(ks.layers.Dense(output_shape, activation='softmax'))

    return model


def run_svhn_test():
    data_loader = dp.DataLoader()
    model_net = construct_net_svhn(data_loader.train_x.shape[1], data_loader.train_y.shape[1])
    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model_net.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model_net.fit(data_loader.train_x, data_loader.train_y, epochs=10, verbose=True)
    loss, accuracy = model_net.evaluate(data_loader.test_x, data_loader.test_y, verbose=False)
    model_net.summary()
    print(loss)
    print(accuracy)
    return

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


def load_mnist():
    image_size = 28  # width and height
    no_of_different_labels = 10  # i.e. 0, 1, 2, 3, ..., 9
    image_pixels = image_size * image_size
    data_path = "../../"
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


def root_mean_squared_error(y_true, y_pred):
    return tf.keras.backend.sqrt(tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true)))


def run_iris_test_keras():
    x_iris, y_iris = ds.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x_iris, y_iris, test_size=0.2)
    targets_train = np.array(y_train, ndmin=2).T
    one_hot_train_y = one_hot(targets_train)
    targets_test = np.array(y_test, ndmin=2).T
    one_hot_test_y = one_hot(targets_test)
    model_net = construct_net(x_train.shape[1], one_hot_train_y.shape[1])
    sgd = tf.keras.optimizers.SGD(lr=0.001)

    model_net.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    model_net.fit(x_train, one_hot_train_y, epochs=200, verbose=True)
    loss, accuracy = model_net.evaluate(x_test, one_hot_test_y, verbose=False)
    model_net.summary()
    print(loss)
    print(accuracy)
    return


def run_mnist_test_keras():
    train_imgs, train_labels_one_hot, test_imgs, test_labels_one_hot = load_mnist_keras()
    model_net = construct_net(784, 10)
    sgd = tf.keras.optimizers.SGD(lr=0.01)
    model_net.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model_net.fit(train_imgs, train_labels_one_hot, epochs=1, verbose=True)
    loss, accuracy = model_net.evaluate(test_imgs, test_labels_one_hot, verbose=False)
    model_net.summary()
    print(loss)
    print(accuracy)
    return

def main():
    #run_mnist_test_keras()
    #run_iris_test_keras()
    run_svhn_test()



if __name__ == '__main__':
    main()








