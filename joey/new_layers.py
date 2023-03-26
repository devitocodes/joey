from functools import reduce

import numpy as np
from devito import Grid, Eq, Inc, Max, Function, exp, sum, Constant, sqrt
from numpy.core.multiarray import array
from scipy.special import softmax

from joey import Layer, default_name_allocator, default_dim_allocator
from joey.funtional import Dropout3d, Softmax3d
from joey.utils import get_tensor_3d


class FullyConnected3d(Layer):
    """
    A Layer subclass corresponding to a full connection (FC) layer.

    Parameters
    ----------
    weight_size : (int, int)
        The shape of a weight matrix (represented internally by a NumPy array)
        expressed as (rows, columns).
    input_size : (int, int)
        The shape of input data expressed as (rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    """

    def __init__(self, weight_size, input_size, name_allocator_func=default_name_allocator,
                 dim_allocator_func=default_dim_allocator, activation=None,
                 generate_code=False):
        super().__init__(weight_size, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code)

    def _allocate(self, weight_size, input_size, name_allocator_func, dim_allocator_func, **kwargs):

        t1, t2, t3, t4 = dim_allocator_func(4)

        self._dimensions = (t1, t2, t3, t4)

        gridW = Grid(shape=weight_size, dimensions=(t4, t3))
        W = Function(name=name_allocator_func(), grid=gridW, space_order=0,
                     dtype=np.float64)

        gridV_dimensions = (t1, t2, t3)
        gridR_dimensions = (t1, t2, t4)
        gridR_shape = (input_size[0], input_size[1], weight_size[0])

        gridV = Grid(shape=input_size, dimensions=gridV_dimensions)
        V = Function(name=name_allocator_func(), grid=gridV, space_order=0,
                     dtype=np.float64)

        gridR = Grid(shape=gridR_shape, dimensions=gridR_dimensions)
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        if self._activation is not None:
            self._T = Function(name=name_allocator_func(), grid=gridR,
                               space_order=0, dtype=np.float64)

        bias_grid = Grid(shape=weight_size[0],
                         dimensions=(t4,))
        bias = Function(name=name_allocator_func(), grid=bias_grid,
                        space_order=0, dtype=np.float64)

        kernel_grad = Function(name=name_allocator_func(),
                               grid=gridW, space_order=0, dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=gridR, space_order=0,
                               dtype=np.float64)

        bias_grad = Function(name=name_allocator_func(),
                             grid=bias_grid, space_order=0, dtype=np.float64)

        return W, V, R, bias, kernel_grad, output_grad, bias_grad

    def execute(self, input_data, bias, weight_data=None):
        if weight_data is not None:
            self._K.data[:] = weight_data

        self._I.data[:] = input_data
        self._bias.data[:] = bias

        if self._activation is not None:
            self._T.data[:] = 0

        self._R.data[:] = 0

        return super().execute()

    def equations(self):
        eqs = [Inc(self._R, self._K * self._I)]

        if self._activation is not None:
            eqs.append(Eq(self._R, self._activation(self._bias + self._R)))
        else:
            eqs.append(Inc(self._R, self._bias))

        return eqs, []

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        t1, t2, _ = layer.result_gradients.dimensions  # t1, t2, t4
        t4, t3 = layer.kernel_gradients.dimensions  # t4, t3

        if prev_layer is None:
            return ([Inc(layer.bias_gradients, layer.result_gradients),
                     Inc(layer.kernel_gradients,
                         layer.input * layer.result_gradients)], [])

        return ([Inc(layer.result_gradients,
                     prev_layer.kernel *
                     prev_layer.result_gradients)] +
                layer.activation.backprop_eqs(layer) +
                [Inc(layer.bias_gradients, layer.result_gradients),
                 Eq(layer.kernel_gradients,
                    layer.kernel_gradients + layer.input * layer.result_gradients)
                 ], [])


class Norm3d(Layer):

    def __init__(self, weight_size, input_size, name_allocator_func=default_name_allocator,
                 dim_allocator_func=default_dim_allocator, activation=None,
                 generate_code=False, **kwargs):
        super().__init__(weight_size, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code, **kwargs)

    def _allocate(self, weight_size, input_size, name_allocator_func, dim_allocator_func, **kwargs):
        batch, row, col, col2 = dim_allocator_func(4)

        self.eps = kwargs.get('eps', 1e-6)
        self._dimensions = (batch, row, col, col2)
        self.shape = input_size

        self.N = weight_size[0]

        gridW = Grid(shape=weight_size, dimensions=(col,))
        W = Function(name=name_allocator_func(), grid=gridW, space_order=0, dtype=np.float64)

        gridV_dimensions = (batch, row, col)
        gridR_dimensions = (batch, row, col)
        gridR_shape = input_size

        gridV = Grid(shape=input_size, dimensions=gridV_dimensions)
        V = Function(name=name_allocator_func(), grid=gridV, space_order=0, dtype=np.float64)

        gridR = Grid(shape=gridR_shape, dimensions=gridR_dimensions)
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0, dtype=np.float64)

        bias_grid = Grid(shape=weight_size[0], dimensions=(col,))
        bias = Function(name=name_allocator_func(), grid=bias_grid, space_order=0, dtype=np.float64)

        kernel_grad = Function(name=name_allocator_func(), grid=gridW, space_order=0, dtype=np.float64)

        output_grad = Function(name=name_allocator_func(), grid=gridR, space_order=0, dtype=np.float64)

        bias_grad = Function(name=name_allocator_func(), grid=bias_grid, space_order=0, dtype=np.float64)

        return W, V, R, bias, kernel_grad, output_grad, bias_grad

    def init_params(self):
        self._K.data[:] = np.ones(self.N)
        self._bias.data[:] = np.zeros(self.N)

    def execute(self, kernel_data=None, input_data=None, bias=None) -> array:
        pass

    def equations(self) -> (list, list):
        batch, row, col, col2 = self._dimensions
        axis = Constant(name=self.name + 'dim_1', value=self._I.shape[-1])
        eps = Constant(name=self.name + 'eps', value=self.eps)

        print(self._I.shape[-1])

        result_sum = get_tensor_3d(default_name_allocator('result_sum_' + self.name),
                                   shape=(self.shape[0:2] + (1,)),
                                   dims=(batch, row, col2))
        result_mean = get_tensor_3d(default_name_allocator('result_mean_' + self.name),
                                    shape=(self.shape[0:2] + (1,)),
                                    dims=(batch, row, col2))
        result_std = get_tensor_3d(default_name_allocator('result_std_' + self.name),
                                   shape=(self.shape[0:2] + (1,)),
                                   dims=(batch, row, col2))

        return [
            Eq(result_sum, 0),
            Inc(result_sum, self._I),
            Eq(result_mean, result_sum / axis),
            Inc(result_std, ((self._I - result_mean) ** 2)),
            Eq(result_std, result_std / axis),
            Eq(result_std, sqrt(result_std)),
            Eq(self._R, self._K * (self._I - result_mean) / (result_std + eps)),
            Inc(self._R, self._bias[col])
        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Norm2:
    def __init__(self, d_model, eps=1e-6):
        super().__init__()

        self.size = d_model

        self.weight = np.ones(self.size)
        self.bias = np.zeros(self.size)
        self.G_weight = np.ones(self.size)
        self.G_bias = np.zeros(self.size)
        self.x = None
        self.result = None
        self.eps = eps

    def forward(self, x: np.ndarray):
        self.x = x
        norm = self.weight * (x - x.mean(axis=-1, keepdims=True)) / (
                x.std(axis=-1, keepdims=True) + self.eps) + self.bias
        self.result = norm
        return norm

    def backward(self, grad: np.ndarray):
        pass

#
# from joey.net import Net
#
# #
# a = FullyConnected3d(input_size=(28, 28, 64), weight_size=(512, 64))
# a1 = FullyConnected3d(input_size=(28, 28, 512), weight_size=(228, 512))
# b = FullyConnected3d(input_size=(28, 28, 228), weight_size=(10, 228))
# drop = Dropout3d(name='dropout_', shape=(28, 28, 10), dropout=0.1)
# sof = Softmax3d(name='softmax', shape=(28, 28, 10))
# norm = Norm3d(input_size=(28, 28, 10), weight_size=(10,), name='norm')
# x = Net([
#     a, a1,  b, drop, sof, norm
# ])
#
# print(x._forward_operator)
# q = x.forward(np.random.rand(28, 28, 64))
# n = Norm2(10)
# normm = n.forward(x._layers[3].result.data)
#
# # soft = softmax(x._layers[2].result.data,axis=-1)
# # print(np.sum(soft, axis=-1))
# # print(np.sum(q, axis=-1))
# print(np.allclose(normm, q, rtol=1e-3, atol=1e-3))
# # print(np.sum(q, axis=-1))
