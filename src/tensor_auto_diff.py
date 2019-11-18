import numpy as np
import math
from auto_diff import Constant, Var


class ConstantTensor:
    def __init__(self, tensor):
        self.value = tensor
        self.gradient = self.function_gradient_constant(x=tensor)

    @staticmethod
    @np.vectorize
    def function_tensor_constant(x):
        return Constant(x)

    @staticmethod
    @np.vectorize
    def function_gradient_constant(x):
        return Constant(0)

    def eval(self):
        return self.value

    def __str__(self):
        return str(self.value)


class VarTensor:
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
    x = Var(name="x",value=2)
    y = Var(name="y", value=4)

    f = 'Div(Constant(1), Add(Constant(1), Exponential(Mul(Constant(-1), {0}))))'
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

