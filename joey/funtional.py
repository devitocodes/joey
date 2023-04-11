from abc import ABC, abstractmethod
from functools import reduce
import numpy as np
from devito import Eq, Function, exp, Inc

from joey import default_name_allocator, Layer, default_dim_allocator
from joey.utils import get_tensor_2d, get_tensor_1d, get_tensor_3d, get_tensor_4d


def kernel_shape(x):
    return reduce(lambda a, b: a * b, x)


class Functional(Layer):

    def _allocate(self, kernel_size, input_size, name_allocator_func, dim_allocator_func, **kwargs) -> (
            Function, Function, Function,
            Function, Function, Function,
            Function):
        return self._K, self._I, self._R, self.bias, self._KG, self._RG, self._biasG

    def execute(self, kernel_data=None, input_data=None, bias=None) -> np.array:
        pass

    @abstractmethod
    def equations(self) -> (list, list):
        pass

    @abstractmethod
    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        pass

    def __init__(self):
        pass


class BaseDropout(Functional):
    def equations(self) -> (list, list):
        pass

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        pass

    def init_params(self):
        K = int(self.N * self.dropout)
        arr = np.array([0] * K + [1] * (self.N - K))
        np.random.shuffle(arr)

        self._K.data[:] = arr.reshape(*self.shape)


class Dropout1d(BaseDropout):
    def __init__(self, name, shape, kernel_size, input_size, dropout=0.1, **kwargs):
        self.N = kernel_shape(shape)
        self.dropout = dropout
        self.name = name
        self.shape = shape
        self.propagate = False

        self._I = get_tensor_1d(default_name_allocator('input_' + self.name), shape=self.shape)
        self._K = get_tensor_1d(default_name_allocator('kernel_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)
        self._R = get_tensor_1d(default_name_allocator('result_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)

        self._bias, self._KG, self._RG, self._biasG = None, None, None, None

    def equations(self) -> (list, list):
        a = self._R.dimensions
        return [
                   Eq(self._R[a], self._I[a] * self._K[a])
               ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Dropout2d(BaseDropout):
    def __init__(self, name, shape, kernel_size, input_size, dropout=0.1, **kwargs):
        self.N = kernel_shape(shape)
        self.dropout = dropout
        self.name = name
        self.shape = shape
        self.propagate = False

        self._I = get_tensor_2d(default_name_allocator('input_' + self.name), shape=self.shape)
        self._K = get_tensor_2d(default_name_allocator('kernel_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)
        self._R = get_tensor_2d(default_name_allocator('result_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)

        self._bias, self._KG, self._RG, self._biasG = None, None, None, None

    def equations(self) -> (list, list):
        a, b = self._R.dimensions
        return [
                   Eq(self._R[a, b], self._I[a, b] * self._K[a, b])
               ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Dropout3d(BaseDropout):
    def __init__(self, name, shape, dropout=0.1, **kwargs):
        self.N = kernel_shape(shape)
        self.dropout = dropout
        self.name = name
        self.shape = shape
        self.propagate = False

        self._I = get_tensor_3d(default_name_allocator('input_' + self.name), shape=self.shape)
        self._K = get_tensor_3d(default_name_allocator('kernel_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)
        self._R = get_tensor_3d(default_name_allocator('result_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)

        self._bias, self._KG, self._RG, self._biasG = None, None, None, None

    def equations(self) -> (list, list):
        a, b, c = self._R.dimensions
        return [
                   Eq(self._R[a, b, c], self._I[a, b, c] * self._K[a, b, c])
               ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Dropout4d(BaseDropout):
    def __init__(self, name, shape, dropout=0.1, **kwargs):
        self.N = kernel_shape(shape)
        self.dropout = dropout
        self.name = name
        self.shape = shape
        self.propagate = False

        self._I = get_tensor_4d(default_name_allocator('input_' + self.name), shape=self.shape)
        self._K = get_tensor_4d(default_name_allocator('kernel_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)
        self._R = get_tensor_4d(default_name_allocator('result_' + self.name), shape=self.shape,
                                dims=self._I.dimensions)

        self._bias, self._KG, self._RG, self._biasG = None, None, None, None

    def equations(self) -> (list, list):
        a, b, c, d = self._R.dimensions
        return [
           Eq(self._R[a, b, c, d], self._I[a, b, c, d] * self._K[a, b, c, d])
        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Softmax3d(Functional):
    def __init__(self, name, shape, **kwargs):
        self.name = name
        self.shape = shape
        self.propagate = False

        self._I = get_tensor_3d(default_name_allocator('input_' + self.name), shape=self.shape)
        self._R = get_tensor_3d(default_name_allocator('result_' + self.name), shape=self.shape)

        self._bias, self._KG, self._RG, self._biasG, self._K = None, None, None, None, None

    def equations(self) -> (list, list):
        a, b, c = self._I.dimensions
        x, y, z = self._R.dimensions

        h = default_dim_allocator(1)[0]
        expon = get_tensor_3d(default_name_allocator('exponential_' + self.name), shape=self.shape, dims=(a, b, c))
        sum_last_axis = get_tensor_3d(default_name_allocator('sum_all_' + self.name), shape=(self.shape[0:2] + (1,)),
                                      dims=(a, b, h))

        return [
            Eq(self.result, 0),
            Eq(expon[a, b, c], exp(self._I[a, b, c])),
            Eq(sum_last_axis[a, b, h], 0),
            Inc(sum_last_axis[a, b, h], expon[a, b, c]),
            Eq(self.result[x, y, z], expon[x, y, z] / sum_last_axis[x, y, h]),
        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []

    def init_params(self):
        pass


class Softmax4d(Functional):
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.propagate = False

        self._I = get_tensor_4d(default_name_allocator('input_' + self.name), shape=self.shape)
        self._R = get_tensor_4d(default_name_allocator('result_' + self.name), shape=self.shape)

        self.dimensions = self._R.dimensions

        self._bias, self._KG, self._RG, self._biasG, self._K = None, None, None, None, None

    def equations(self) -> (list, list):
        a, b, c, d = self._I.dimensions
        x, y, z, w = self._R.dimensions

        h = default_dim_allocator(1)[0]
        expon = get_tensor_4d(default_name_allocator('exponential_' + self.name), shape=self.shape, dims=(a, b, c, d))
        sum_last_axis = get_tensor_4d(default_name_allocator('sum_all_' + self.name), shape=(self.shape[0:3] + (1,)),
                                      dims=(a, b, c, h))
        eqs = [Eq(self.result, 0)]
        eqs += [
            Eq(expon[a, b, c, d], exp(self.input[a, b, c, d])),
            Eq(sum_last_axis[a, b, c, h], 0),
            Inc(sum_last_axis[a, b, c, h], expon[a, b, c, d]),
            Eq(self.result[x, y, z, w], expon[x, y, z, w] / sum_last_axis[x, y, z, h]),

        ]

        return eqs, []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []

    def init_params(self):
        pass


class Expand3to4(Functional):

    def __init__(self, name, shape_in, shape_out):
        assert shape_in[-1] == (shape_out[-1] * shape_out[-2]), 'The last Input dimension must match the ' \
                                                                'multiplication ' \
                                                   'of the 2 last Result dimensions.'

        self.name = name
        self.propagate = False

        self._I = get_tensor_3d(default_name_allocator('input_' + self.name), shape=shape_in)
        self._R = get_tensor_4d(default_name_allocator('result_' + self.name), shape=shape_out)

        self.dimensions = self._R.dimensions

        self._bias, self._KG, self._RG, self._biasG, self._K = None, None, None, None, None

    def init_params(self):
        pass

    def equations(self) -> (list, list):
        a, b, c, d = self._R.dimensions
        _, _, D = self._I.shape
        return [
           Eq(self._R[a, b, c, d], self._I[a, b, (c * D) + d])
        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Contract4to3(Functional):

    def __init__(self, name, shape_in, shape_out):
        assert shape_out[-1] == (shape_in[-1] * shape_in[-2]), 'The last Result dimension must match the ' \
                                                                'multiplication ' \
                                                                'of the 2 last Input dimensions.'

        self.name = name
        self.propagate = False

        self._I = get_tensor_4d(default_name_allocator('input_' + self.name), shape=shape_in)
        self._R = get_tensor_3d(default_name_allocator('result_' + self.name), shape=shape_out)

        self.dimensions = self._R.dimensions

        self._bias, self._KG, self._RG, self._biasG, self._K = None, None, None, None, None

    def init_params(self):
        pass

    def equations(self) -> (list, list):
        a, b, c, d = self._I.dimensions
        _, _, _, D = self._I.shape
        return [
           Eq(self._R[a, b, (c * D + d)], self._I[a, b, c, d])
        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Reduce2ndDimension3d(Functional):

    def __init__(self, name, shape):

        self._I = get_tensor_3d('reduce_input_' + name, shape=shape)
        self._R = get_tensor_2d('reduce_result_' + name, shape=(shape[0], shape[-1]))

        self._bias, self._KG, self._RG, self._biasG, self._K = None, None, None, None, None

    def equations(self) -> (list, list):
        a, b = self.result.dimensions
        _, X, _ = self.input.shape
        return [
            Eq(self._R[a, b], self._I[a, X-1, b])

        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []
