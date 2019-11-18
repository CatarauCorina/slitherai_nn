import numpy as np
import math
from functools import reduce


class Constant:
    def __init__(self, value):
        self.value = value
        self.gradient = lambda var: Constant(0)

    def eval(self):
        return self.value

    def __str__(self):
        return str(self.value)


class Var:
    def __init__(self, name, value=0):
        self.value = value
        self.name = name
        self.gradient = lambda var: Constant(1) if var is self else Constant(0)

    def eval(self):
        return self.value

    def __str__(self):
        return str(self.name)


class Add:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.gradient = lambda var: Add(self.a.gradient(var), self.b.gradient(var))

    def eval(self):
        return self.a.eval() + self.b.eval()

    def __str__(self):
        return "({} + {})".format(self.a, self.b)


class Sub:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.gradient = lambda var: Sub(self.a.gradient(var), self.b.gradient(var))

    def eval(self):
        return self.a.eval() - self.b.eval()

    def __str__(self):
        return "({} - {})".format(self.a, self.b)


class Mul:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.gradient = lambda var: Add(Mul(self.a.gradient(var), self.b), Mul(self.a, self.b.gradient(var)))

    def eval(self):
        return self.a.eval() * self.b.eval()

    def __str__(self):
        return "({} * {})".format(self.a, self.b)


class Div:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.gradient = lambda var: Div(
            Sub(Mul(self.a.gradient(var), self.b), Mul(self.a, self.b.gradient(var))),
            Mul(self.b, self.b)
        )

    def eval(self):
        return self.a.eval() / self.b.eval()

    def __str__(self):
        return "({} / {})".format(self.a, self.b)


class Pow:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.gradient = lambda var: Mul(self.b, Pow(self.a, Sub(self.b, 1)))

    def eval(self):
        return self.a.eval() ** self.b.eval()

    def __str__(self):
        return "({} ^ {})".format(self.a, self.b)


class Exponential:
    def __init__(self, a):
        self.a = a
        self.gradient = lambda var: Mul(self.a.gradient(var), Exponential(self.a))

    def eval(self):

        return np.exp(self.a.eval())

    def __str__(self):
        return "(e ^ {})".format(self.a)


class Log:
    def __init__(self, a):
        self.a = a
        self.gradient = lambda var: Div(self.a.gradient(var), var)

    def eval(self):

        return np.log(self.a.eval())

    def __str__(self):
        return "(log {})".format(self.a)



class Max:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.gradient = lambda var: self.a.gradient(var) if self.a.eval() > self.b.eval() else self.b.gradient(var)

    def eval(self):
        return self.a.eval() if self.a.eval() > self.b.eval() else self.b.eval()

    def __str__(self):
        return "(max({},{}))".format(self.a, self.b)


def main():
    x = Var(name="x", value=2)
    y = Var(name="y", value=4)
    x_arr = [2.0, 1.0, 0.1]
    t = [1, 0, 0]
    t_vals = [Var(name=f't_{i}', value=float(t[i])) for i in range(len(t))]
    x_vals = [Var(name=f's_{i}', value=float(x_arr[i])) for i in range(len(x_arr))]
    log_x = [Log(x_val) for x_val in x_vals]
    prod_ce = [Mul(ti, logxi) for ti, logxi in zip(t_vals,log_x)]
    sum_ce = reduce(lambda x, y: Add(x, y), prod_ce)
    sum_ce = Mul(Constant(-1), sum_ce)

    x_softmax = [1, 2]
    x_arr_var = [Var(name=f'x_{i}', value=float(x_softmax[i])) for i in range(len(x_softmax))]
    exp_arr = [Exponential(v) for v in x_arr_var]
    sum_den = reduce(lambda x, y: Add(x, y), exp_arr)
    # not_i = x_arr[1:]
    # not_i_e = np.exp(not_i)
    # not_i_e_sum = np.sum(not_i_e)
    # x_v = Var(name='x', value=x_arr[0])
    # var_not_i_e = Var(name='y', value=float(not_i_e_sum))
    f = 'Div(Constant(1), Add(Constant(1), Exponential(Mul(Constant(-1), {0}))))'
    softmax = 'Div({0},{1})'
    softmax_den = ''
    res = eval(softmax.format(exp_arr[0], sum_den)).eval()
    res_gradient = Div(exp_arr[0], sum_den).gradient(x_arr_var[0]).eval()
    g = Add(Add(Mul(Mul(x, x), y), y), Constant(2)) # f(x,y) = x²y + y + 2
    z = 'Add(Mul(Var(name="x",value={0}), Var(name="x",value={0})), Constant(2))'
    h = Add(Constant(1), Exponential(Mul(Constant(-1), x)))
    # z = Div(Mul(Constant(2), x), Add(Mul(Constant(3), x), Constant(1)))
    m = 'Max({0}, Constant(0))'
    x_arr = np.array([-1, 2, -3, 4])
    v_relu = [eval(z.format(x)).gradient(Var(name='x',value=x)).eval() for x in x_arr]
    print(v_relu)
    print(g.__str__())
    # print(m.eval())
    print(g.gradient(x).eval())
    print(g.gradient(y).eval())

    x_arr_var = [Var(name=x, value=v) for v in x_arr]
    slope = 1
    relu_derivative = "1 if x > 0 else 0"
    sigmoid_derivative = f'(1/1+{np.e}**(-{slope}*x))*(1-(1/1+{np.e}**(-{slope}*x)))'
    v_relu = [eval(m.format(x)).eval() for x in x_arr_var]
    v_relu_non_auto = np.maximum(0, x_arr)
    print(v_relu)
    print(v_relu_non_auto)
    print('-------')
    v = [eval(f.format(x)).gradient(x).eval() for x in x_arr_var]
    derivative_output = [eval(sigmoid_derivative) for x in x_arr]
    #
    print(v)
    print(derivative_output)
    # print(f.gradient(y).eval())
    # print(derivative_output)
    # print(f.gradient(Var(name='x',value=2)).eval())
    x=-1
    print(eval(sigmoid_derivative))
    # print(m.gradient(x).eval())

    return

if __name__ == '__main__':
    main()

