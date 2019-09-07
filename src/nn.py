import sys
import pandas as pd
import numpy as np
import sklearn


class ActivationFunction:
    CHOICES = ['tanh', 'sigmoid', 'relu', 'identity']
    function = None
    output = None

    def __init__(self, x, name=CHOICES[0]):
        self.function = getattr(np, name, None)
        if self.function is None:
            self.function = getattr(self, name, None)
        self.output = self.function(x)

    @staticmethod
    def identity(x, *args, **kwargs):
        return x


class BaseLayer:
    nr_neurons = 0
    sample_size = 0
    data = np.matrix(0)
    activation_type = 'identity'
    prev_layer = None
    activation_function = None
    out = None

    def __init__(self, nr_neurons, sample_size, data, prev_layer=None, activation_type='identity'):
        self.nr_neurons = nr_neurons
        self.sample_size = sample_size
        self.data = data
        self.prev_layer = prev_layer
        self.activation_type = activation_type

    def output(self):
        if not self.prev_layer:
            return self.data
        self.out = ActivationFunction(np.matmul(self.prev_layer.output(), self.data), self.activation_type).output
        return self.out


class InputLayer(BaseLayer):

    def __init__(self, data_x):
        nr_of_rows, nr_of_features = pd.DataFrame(data_x).shape
        BaseLayer.__init__(self, nr_of_features, nr_of_rows, data_x)


class HiddenLayer(BaseLayer):
    init_type = ['random']

    def __init__(self, size, prev_layer, init_type='random'):
        data = getattr(self, f'{init_type}_init')(prev_layer.nr_neurons, size)
        BaseLayer.__init__(self, size, prev_layer.nr_neurons, data, prev_layer, 'tanh')

    def random_init(self, size, nr_neurons):
        return np.random.rand(size, nr_neurons)


class OutputLayer(HiddenLayer):

    def __init__(self, data_y, prev_layer):
        nr_of_rows, nr_of_features = pd.DataFrame(data_y).shape
        HiddenLayer.__init__(self, nr_of_features, prev_layer)


class Network:
    input_data = None
    output_data = None
    crt_layer = None
    layers = []

    def __init__(self, x, y):
        self.input_data = x
        self.output_data = y

    def init_network(self):
        layer_in = InputLayer(self.input_data)
        self.crt_layer = layer_in
        self.layers.append(layer_in)
        return self

    def add_layer(self, size=2, init_type='random', type_layer='Hidden'):
        layer = getattr(sys.modules[__name__], f'{type_layer}Layer')(size, self.crt_layer, init_type)
        self.crt_layer = layer
        self.layers.append(layer)
        return self

    def run(self):
        layer_out = OutputLayer(self.output_data, self.crt_layer)
        self.crt_layer = layer_out
        self.layers.append(layer_out)
        return self.crt_layer.output()





