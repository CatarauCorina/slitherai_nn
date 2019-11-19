import pandas as pd
import numpy as np
import auto_diff as auto_diff
import sklearn
from sympy import *
from scipy.stats import truncnorm
from activations import ActivationFunction, CostFunction


class BaseLayer:

    def __init__(self,
                 nr_neurons=0,
                 sample_size=0,
                 data=np.matrix(0),
                 prev_layer=None,
                 activation_type='sigmoid',
                 out=None,
                 bias=False,
                 bias_data=None,
                 output_layer=False,
                 next_layer=None
                 ):
        self.nr_neurons = nr_neurons
        self.sample_size = sample_size
        self.data = data
        self.prev_layer = prev_layer
        self.activation_type = activation_type
        self.derivative = None
        self.out = out
        self.z_out = None
        self.bias = bias
        self.bias_data = np.zeros(self.nr_neurons)
        self.output_layer = output_layer
        self.crt_err_gradient = None
        self.chain_gradient = None
        self.bias_gradient = None
        self.next_layer = next_layer
        return

    def compute_forward_pass(self):
        if not self.prev_layer:
            self.out = self.data
            # print('----------')
            # print(self)
            # print(self.out)
            # print('----------')

            return self.data
        else:
            self.z_out = np.dot(self.prev_layer.compute_forward_pass(), self.data.T) + self.bias_data
            self.out = ActivationFunction(self.z_out, self.activation_type, layer=self).output
            # print('----------')
            # print(self)
            # print(self.data)
            # print(self.out)
            # print('----------')
            return self.out

    def compute_error_gradient(self):
        cost = CostFunction(self.out, self.target_values, name=self.cost_function)
        if self.cost_function == 'categorical_cross_entropy':
            errors = cost.derivative_cost
        else:
            errors = [eval(cost.derivative_cost) for (predicted, target) in zip(self.out, self.target_values)]
        return np.array(errors, ndmin=2)

    def update_weights(self, learning_rate=0.0001):
        self.data = self.data - learning_rate*self.crt_err_gradient
        if self.bias_gradient is not None:
            self.bias_data = self.bias_data - learning_rate*self.bias_gradient
        return

    def compute_bias_gradient(self, error_gradient):
        shape_grad = error_gradient.shape
        if shape_grad[0] > 0:
            bias_gradient = np.mean(error_gradient, axis=0, keepdims=True)
            return bias_gradient
        return error_gradient

    def compute_backward_pass(self, learning_rate=0.01):
        if self.activation_type == 'softmax':
            derivative_output = np.array(self.derivative, ndmin=2).reshape(self.out.shape)
        else:
            vars = [auto_diff.Var(name="x", value=x) for x in flatten(self.z_out)]
            derivative_output = np.array(
                [eval(self.derivative.format(x)).eval() for x in vars],
                ndmin=2
            ).reshape(self.out.shape)

        if self.next_layer is None:
            error_gradient_out = self.compute_error_gradient() * derivative_output
            self.crt_err_gradient = np.dot(error_gradient_out.T, self.prev_layer.out)
            self.chain_gradient = np.dot(error_gradient_out, self.data)
            self.update_weights(learning_rate)
            self.prev_layer.compute_backward_pass()
        elif self.prev_layer is not None and self.prev_layer.prev_layer is not None:
            error_gradient_hidden = self.next_layer.chain_gradient * derivative_output
            self.bias_gradient = self.compute_bias_gradient(error_gradient_hidden)
            self.crt_err_gradient = np.dot(error_gradient_hidden.T, self.prev_layer.out)
            self.chain_gradient = np.dot(error_gradient_hidden, self.data)
            self.update_weights(learning_rate)
            self.prev_layer.compute_backward_pass()
        elif self.prev_layer.prev_layer is None:
            error_gradient_hidden = self.next_layer.chain_gradient * derivative_output
            self.bias_gradient = self.compute_bias_gradient(error_gradient_hidden)
            self.crt_err_gradient = np.dot(error_gradient_hidden.T, self.prev_layer.out)
            self.update_weights(learning_rate)
            self.chain_gradient = error_gradient_hidden
        else:
            return self.data


class InputLayer(BaseLayer):

    def __init__(self, data_x, bias, shape=None, activation_func='relu'):
        if data_x is None:
            nr_of_rows, nr_of_features = shape
            BaseLayer.__init__(self, nr_of_features, nr_of_rows, bias=bias, activation_type=activation_func)
        else:
            nr_of_rows, nr_of_features = pd.DataFrame(data_x).shape
            BaseLayer.__init__(self, nr_of_features, nr_of_rows, data_x, bias=bias, activation_type=activation_func)
        return


class HiddenLayer(BaseLayer):
    init_type = ['random']

    def __init__(self, size, prev_layer, activation, init_type='debug', bias=True, output_layer=False):
        data = getattr(self, f'{init_type}_init')(prev_layer.nr_neurons, size)
        BaseLayer.__init__(self, size, prev_layer.nr_neurons, data, prev_layer, activation_type=activation, output_layer=output_layer)
        if bias:
            self.bias_data = self.bias_init()

    def truncated_normal(self, mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def debug_init(self, prev_layer_neurons, nr_neurons):
        arr = np.arange(prev_layer_neurons*nr_neurons)
        return arr.reshape(nr_neurons, prev_layer_neurons)

    def random_init_2(self, prev_layer_neurons, nr_neurons):
        x = self.truncated_normal(
            mean=0,
            sd=prev_layer_neurons ** (-1 / 2),
            low=-0.5, upp=0.5
        )
        random_vector = x.rvs((nr_neurons, prev_layer_neurons))
        return random_vector

    def random_init(self, prev_layer_neurons, nr_neurons):
        return np.random.randn(nr_neurons, prev_layer_neurons)*np.sqrt(2/nr_neurons)

    def xavier_init(self, prev_layer_neurons, nr_neurons):
        return np.random.randn(nr_neurons, prev_layer_neurons) * np.sqrt(1 / (nr_neurons + prev_layer_neurons))

    def bias_init(self):
        bias_vector = np.full((1, self.nr_neurons), 0.1)
        return bias_vector


class OutputLayer(HiddenLayer):
    target_values = None

    def __init__(self, data_y, prev_layer, bias, shape=None, activation='relu', cost_function='sse', init_type='random'):
        if data_y is None:
            nr_of_rows, nr_of_features = shape
            self.cost_function = cost_function
            HiddenLayer.__init__(self, nr_of_features, prev_layer, bias=bias, activation=activation, output_layer=True, init_type=init_type)
        else:
            nr_of_rows, nr_of_features = pd.DataFrame(data_y).shape
            self.cost_function = cost_function
            HiddenLayer.__init__(self, nr_of_features, prev_layer, bias=bias, activation=activation, output_layer=True, init_type=init_type)

        self.next_layer = None
        return
