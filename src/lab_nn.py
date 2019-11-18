import os
import numpy as np
import sklearn.datasets as ds
from sklearn.model_selection import train_test_split



@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)


activation_function = sigmoid
from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd,
                     loc=mean,
                     scale=sd)


class NeuralNetwork:
    # intialization method (constructor)
    def __init__(self,
                 no_of_in_nodes,
                 no_of_out_nodes,
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate
        self.create_weight_matrices()

    def create_weight_matrices(self):
        """ 
        A method to initialize the weight 
        matrices of the neural network
        """
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0,
                             sd=1,
                             low=-rad,
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes,
                          self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.who = X.rvs((self.no_of_out_nodes,
                          self.no_of_hidden_nodes))

    def train(self, input_vector, target_vector):
        """
        input_vector and target_vector can 
        be tuple, list or ndarray
        """

        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T

        # first layer
        output_vector1 = np.dot(self.wih, input_vector)

        output_hidden = activation_function(output_vector1)  # output of the hidden layer

        # second (last) layer
        output_vector2 = np.dot(self.who, output_hidden)
        output_network = activation_function(output_vector2)

        # calclate the loss 
        output_errors = target_vector - output_network

        # update the weights:
        derivative_output = output_network * (1.0 - output_network)
        tmp_1 = output_errors * derivative_output
        tmp_1 = self.learning_rate * np.dot(tmp_1, output_hidden.T)

        # calculate hidden errors:
        err = derivative_output*output_errors
        hidden_errors = np.dot(self.who.T, err)
        # update the weights:
        derivative_output = output_hidden * (1.0 - output_hidden)
        tmp_2 = hidden_errors * derivative_output
        print(self.who)
        self.who += tmp_1
        print(self.who)
        print(self.wih)
        self.wih += self.learning_rate * np.dot(tmp_2, input_vector.T)
        print(self.wih)

    def run(self, input_vector):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        # 1st layer
        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        # 2nd layer
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)

        return output_vector

    def confusion_matrix(self, data_array, labels):
        cm = np.zeros((10, 10), int)
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            cm[res_max, int(target)] += 1
        return cm

    def precision(self, label, confusion_matrix):
        col = confusion_matrix[:, label]
        return confusion_matrix[label, label] / col.sum()

    def recall(self, label, confusion_matrix):
        row = confusion_matrix[label, :]
        return confusion_matrix[label, label] / row.sum()

    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            res_max_labels = labels[i].argmax()
            if res_max == res_max_labels:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs


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


def iris():
    x_iris, y_iris = ds.load_iris(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x_iris, y_iris, test_size=0.2)
    targets_train = np.array(y_train, ndmin=2).T
    one_hot_train_y = one_hot(targets_train)
    targets_test = np.array(y_test, ndmin=2).T
    one_hot_test_y = one_hot(targets_test)
    return x_train, one_hot_train_y, x_test, one_hot_test_y


def main():
    x_train_iris, y_train_iris, x_test_iris, y_test_iris = iris()
    image_size = 28
    image_pixels = image_size * image_size
    train_imgs, train_labels_one_hot, test_imgs, test_labels_one_hot = load_mnist()
    ANN = NeuralNetwork(no_of_in_nodes=image_pixels,
                        no_of_out_nodes=10,
                        no_of_hidden_nodes=100,
                        learning_rate=0.1)

    for i in range(len(train_imgs[:1000])):
        print(i)
        ANN.train(train_imgs[i], train_labels_one_hot[i])

    corrects, wrongs = ANN.evaluate(train_imgs[:1000], train_labels_one_hot[:1000])
    print("accruracy train: ", corrects / (corrects + wrongs))
    corrects, wrongs = ANN.evaluate(test_imgs[:100], test_labels_one_hot[:100])
    print("accruracy: test", corrects / (corrects + wrongs))

if __name__ == '__main__':
    main()