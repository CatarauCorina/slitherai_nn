import sys
import pandas as pd
import time
import numpy as np
from tqdm import tqdm
import auto_diff as auto_diff
import sklearn
from sympy import *
from scipy.stats import truncnorm
from layers import BaseLayer, InputLayer, OutputLayer, HiddenLayer
from activations import CostFunction
import utils


class Network:

    def __init__(self,
                 input_data=None,
                 output_data=None,
                 crt_layer=None,
                 learning_rate=0.1,
                 layers=[],
                 bias=False,
                 cost_function='sse',
                 shape_in=None,
                 regularization=None,
                 ):
        self.input_data = input_data
        self.output_data = output_data
        self.crt_layer = crt_layer
        self.learning_rate = learning_rate
        self.layers = layers
        self.bias = bias
        self.network_out = None
        self.cost_function = cost_function
        self.grad = {}
        self.shape_in = shape_in
        self.regularization = regularization
        return

    def init_network(self, activation_in='relu', alpha_regularization=0):
        layer_in = InputLayer(self.input_data, self.bias, self.shape_in, activation_in,
                              alpha_regularization,
                              regularization_type=self.regularization)
        self.crt_layer = layer_in
        self.layers.append(layer_in)
        return self

    def add_layer(self, size=2, init_type='random', type_layer='Hidden', activation='relu',
                  alpha_regularization=0, layer=None):
        if layer is None:
            layer = getattr(sys.modules[__name__], f'{type_layer}Layer')(size, self.crt_layer,
                                                                         activation,
                                                                         init_type,
                                                                         self.bias,
                                                                         regularization_type=self.regularization,
                                                                         alpha_regularization=alpha_regularization)
        else:
            layer.prev_layer = self.crt_layer
        self.crt_layer.next_layer = layer
        self.crt_layer = layer
        self.layers.append(layer)

        return self

    def add_output(self, target_shape, activation='sigmoid',
                   cost_function='sse',
                   init_type='random',
                   alpha_regularization=0):
        layer_out = OutputLayer(
            None,
            self.crt_layer, self.bias,
            shape=target_shape,
            activation=activation,
            cost_function=cost_function,
            regularization_type=self.regularization,
            alpha_regularization=alpha_regularization
        )
        self.crt_layer.next_layer = layer_out
        self.crt_layer = layer_out
        self.layers.append(layer_out)
        self.cost_function = cost_function
        return self

    def calculate_errors(self, computed):
        return self.output_data - computed

    def chain_rule(self):
        cost = CostFunction(self.network_out, self.output_data, name=self.cost_function)
        errors = [eval(cost.derivative_cost) for (predicted, target) in zip(self.network_out, self.output_data)]
        self.grad['error'] = {'out': errors}

    def run_batch(self, data, targets, batch_size=1, eval=False):
        batches_input = np.array_split(data, batch_size)
        targets_input = np.array_split(targets, batch_size)
        out_net = []
        i = 0
        total = min(len(batches_input), len(targets_input))

        with tqdm(total=total) as pbar:
            for (batch_in, target_in) in zip(batches_input, targets_input):
                if eval:
                    out = self.run_forward(batch_in, target_in)
                    out_net.append(out)
                else:
                    self.train(batch_in, target_in)
                i += 1
                pbar.update(1)
        return out_net

    def run_forward(self, data_x, target):
        self.layers[0].data = data_x
        self.layers[len(self.layers)-1].target_values = target

        network_output = self.crt_layer.compute_forward_pass()
        return network_output

    def train(self, data_x, target, learning_rate=0.001):
        self.run_forward(data_x, target)
        self.crt_layer.compute_backward_pass(learning_rate)
        return

    def train_network(self, data_train, targets_train,
                      data_test, targets_test,
                      batch_size=1, nr_epochs=10,
                      online=True
                      ):

        for epoch in range(nr_epochs):
            print("epoch: ", epoch)
            batch_size_train = batch_size
            if online:
                batch_size_train = len(data_train)
            self.run_batch(data_train, targets_train, batch_size_train)
            for lay in self.layers:
                print('-----')
                print(lay.crt_err_gradient)
                print('-----')
            print(self.layers[len(self.layers)-1].out)
            print("Eval-train")
            corrects, wrongs = self.evaluate(data_train, targets_train)
            print("accuracy train: ", (corrects / (corrects + wrongs))*100)
            print("Eval-test")
            corrects, wrongs = self.evaluate(data_test, targets_test)
            print("accuracy: test", (corrects / (corrects + wrongs))*100)

    def evaluate(self, test_input, test_output):
        out = self.run_batch(test_input, test_output, batch_size=len(test_input), eval=True)
        corrects, wrongs = 0, 0
        for i in range(len(test_output)):
            res_max = out[i].argmax()
            res_max_target = test_output[i].argmax()
            if res_max == res_max_target:
                corrects += 1
            else:
                wrongs += 1

        return corrects, wrongs







