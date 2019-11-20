import sys
import pandas as pd
import numpy as np
from math import log
import auto_diff as auto_diff
import sklearn
from sympy import flatten
from functools import reduce
from scipy.stats import truncnorm


class ActivationFunction:
    CHOICES = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'identity', 'softmax', 'threshold']
    function = None
    output = None
    gradient = None

    def __init__(self, x_arr=None, name=CHOICES[0], layer=None):
        self.function = getattr(self, name, None)
        if name == 'softmax':
            self.softmax_batch(x_arr, layer)
        else:
            self.derivative = None
            if x_arr is not None:
                vars = [auto_diff.Var(name="x", value=x) for x in flatten(x_arr)]
                self.output = np.array(
                    [eval(self.function().format(x)).eval() for x in vars],
                    ndmin=2
                ).reshape(x_arr.shape)
                layer.derivative = self.gradient
        return

    def softmax_batch(self, x_arr, layer):
        nr_values = x_arr.shape[0]
        if nr_values == 1:
            softmax_sample = self.softmax_sample(x_arr)
            output = np.array(
                softmax_sample['out'],
                ndmin=2
            ).reshape(x_arr.shape)
            self.output = output
            layer.derivative = np.array(softmax_sample['deriv']).reshape(x_arr.shape)
        else:
            lambda_soft = [self.softmax_sample(np.array(x, ndmin=2)) for x in x_arr]
            lambda_result = pd.DataFrame(lambda_soft)
            self.output = np.array(lambda_result['out'].to_list()).reshape(x_arr.shape)
            layer.derivative = np.array(lambda_result['deriv'].to_list()).reshape(x_arr.shape)

        return

    def softmax_sample(self, x_arr):
        x_flatten = flatten(x_arr)
        stable_softmax = self.softmax_stabilizer(x_flatten)
        x_arr_var = [auto_diff.Var(name=f'x_{i}', value=float(stable_softmax[i])) for i in range(len(stable_softmax))]
        exp_arr = [auto_diff.Exponential(var) for var in x_arr_var]
        sum_denominator = reduce(lambda x, y: auto_diff.Add(x, y), exp_arr)
        softmax_results = [self.function(x, sum_denominator, activation) for (x, activation) in zip(exp_arr, x_arr_var)]
        softmax_forward = [softmax_result[0] for softmax_result in softmax_results]
        softmax_back = [softmax_result[1] for softmax_result in softmax_results]

        return {'out': softmax_forward, 'deriv': softmax_back}

    def softmax_stabilizer(self, x_arr):
        max_x_arr = np.max(x_arr)
        return x_arr - max_x_arr

    def identity(self, value=0):
        identity_func = '{0}'
        self.gradient = '{0}.gradient({0})'
        return identity_func

    def threshold(self, x_arr, th=0):
        self.derivative = ""
        return [x >= th for x in x_arr]

    def sigmoid(self, value=0):
        x = auto_diff.Var(name='x')
        sigmoid_func = 'auto_diff.Div(auto_diff.Constant(1),' \
                       'auto_diff.Add(auto_diff.Constant(1), ' \
                       'auto_diff.Exponential(auto_diff.Mul(auto_diff.Constant(-1), {0}))))'
        self.gradient = f'{sigmoid_func}.gradient({{0}})'
        return sigmoid_func

    def softmax(self, val, sum_denominator, activation_derivative):
        softmax_func = auto_diff.Div(val, sum_denominator).eval()
        softmax_result_gradient = auto_diff.Div(val, sum_denominator).gradient(activation_derivative).eval()
        return softmax_func, softmax_result_gradient

    def relu(self, value=0):
        relu_func = 'auto_diff.Max({0}, auto_diff.Constant(0))'
        self.gradient = 'auto_diff.Max({0}, auto_diff.Constant(0)).gradient({0})'
        return relu_func


class CostFunction:
    CHOICES = ['sse']

    def __init__(self, predicted_arr=None, target_arr=None, name=CHOICES[0]):
        self.cost_function = getattr(self, name, None)
        self.derivative_cost = None
        if predicted_arr is not None and target_arr is not None:
            self.output = self.cost_function(predicted_arr, target_arr)
        return

    def sse(self, predicted, target):
        self.derivative_cost = f'target-predicted'
        return 1/2*np.sum((target-predicted)**2)

    def categorical_cross_entropy(self, predicted, target):
        shape = target.shape[0]
        if shape == 1:
            cost = self.categorical_cross_entropy_sample(predicted, target)
            self.derivative_cost = cost['derivative_error']
            return cost['error']
        else:
            lambda_ce = [self.categorical_cross_entropy_sample(predicted[i], target[i]) for i in range(len(target))]
            lambda_result = pd.DataFrame(lambda_ce)
            output = (np.array(lambda_result['error'].to_list()).sum())/len(target)
            self.derivative_cost = np.mean(np.array(lambda_result['derivative_error'].to_list()), axis=0, keepdims=True)
            return output

    def categorical_cross_entropy_sample(self, predicted, target):
        target_flatten = flatten(target)
        predicted_flatten = flatten(predicted)
        target_var = [auto_diff.Var(name=f't_{i}', value=float(target_flatten[i]))
                      for i in range(len(target_flatten))]
        predicted_var = [auto_diff.Var(name=f's_{i}', value=np.maximum(1e-15, float(predicted_flatten[i])))
                         for i in range(len(predicted_flatten))]
        log_predicted_var = [auto_diff.Log(pred_var) for pred_var in predicted_var]
        prod_target_predicted = [auto_diff.Mul(ti, log_pred_i) for ti, log_pred_i in zip(target_var, log_predicted_var)]
        sum_ce = reduce(lambda x, y: auto_diff.Add(x, y), prod_target_predicted)
        sum_ce = auto_diff.Mul(auto_diff.Constant(-1), sum_ce)
        cost = predicted - target

        return {'error': sum_ce.eval(), 'derivative_error': cost}
