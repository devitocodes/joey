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

        return self._K, self._I, self._R, None, None, None, None

    def execute(self, kernel_data=None, input_data=None, bias=None) -> np.array:
        pass

    @abstractmethod
    def equations(self) -> (list, list):
        pass

    @abstractmethod
    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        pass

    def __init__(self, **kwargs):
        pass


class Dropout2d(Functional):

    def __init__(self, name, shape, dropout=0.1):
        super().__init__()

        self.N = kernel_shape(shape)
        self.dropout = dropout
        self.name = name
        self.shape = shape
        self.INPUT_EQ = input
        self.init_params = False
        self.propagate = False
        self.dims = self.INPUT_EQ.dimensions

        self._K = get_tensor_1d(default_name_allocator('kernel_' + self.name), shape=(self.N,))

        self._I = self.INPUT_EQ
        self._R = self._allocate()

    def _allocate(self, **kwargs) -> (Function, Function, Function,
                                      Function, Function, Function,
                                      Function):
        return get_tensor_2d(default_name_allocator('result_' + self.name), shape=self.shape, dims=self.dims)

    def equations(self) -> (list, list):
        a, b = self.RESULT_EQ.dimensions
        return [
            Eq(self.RESULT_EQ, self.INPUT_EQ * self.KERNEL_EQ[a * b])
        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        pass

    def init_params(self):
        K = int(self.N * self.dropout)
        arr = np.array([0] * K + [1] * (self.N - K))
        np.random.shuffle(arr)

        self._K.data[:] = arr


class Dropout3d(Functional):
    def __init__(self, name, shape, dropout=0.1):
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

    def init_params(self):
        K = int(self.N * self.dropout)
        arr = np.array([0] * K + [1] * (self.N - K))
        np.random.shuffle(arr)

        self._K.data[:] = arr.reshape(*self.shape)


class Dropout4d(Functional):
    def __init__(self, name, shape, input: Function, dropout=0.1):
        super().__init__(shape, input)
        self.N = kernel_shape(shape)
        self.dropout = dropout
        self.name = name
        self.shape = shape
        self.INPUT_EQ = input
        self.dims = self.INPUT_EQ.dimensions

    def _allocate(self, **kwargs) -> (Function, Function, Function,
                                      Function, Function, Function,
                                      Function):
        self.RESULT_EQ = get_tensor_4d(default_name_allocator('result_' + self.name), shape=self.shape, dims=self.dims)
        self.KERNEL_EQ = get_tensor_1d(default_name_allocator('kernel_' + self.name), shape=(self.N,))

        K = int(self.N * self.dropout)
        arr = np.array([0] * K + [1] * (self.N - K))
        np.random.shuffle(arr)

        self.KERNEL_EQ.data[:] = arr

    def execute(self, kernel_data=None, input_data=None, bias=None) -> np.array:
        return super().execute()

    def equations(self) -> (list, list):
        a, b, c, d = self.RESULT_EQ.dimensions
        return [
            Eq(self.RESULT_EQ, self.INPUT_EQ * self.KERNEL_EQ[a * b * c * d])
        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []


class Softmax3d(Functional):
    def __init__(self, name, shape):
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
            Eq(expon[a, b, c], exp(self._I[a, b, c])),
            Eq(sum_last_axis[a, b, h], 0),
            Inc(sum_last_axis[a, b, h], expon[a, b, c]),
            Eq(self._R[x, y, z], expon[x, y, z] / sum_last_axis[x, y, h]),

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

        self._bias, self._KG, self._RG, self._biasG, self._K = None, None, None, None, None

    def equations(self) -> (list, list):
        a, b, c, d = self._I.dimensions
        x, y, z, w = self._R.dimensions

        h = default_dim_allocator(1)[0]
        expon = get_tensor_4d(default_name_allocator('exponential_' + self.name), shape=self.shape, dims=(a, b, c, d))
        sum_last_axis = get_tensor_4d(default_name_allocator('sum_all_' + self.name), shape=(self.shape[0:3] + (1,)),
                                      dims=(a, b, c, h))

        return [
            Eq(expon[a, b, c, d], exp(self._I[a, b, c, d])),
            Eq(sum_last_axis[a, b, c, h], 0),
            Inc(sum_last_axis[a, b, c, h], expon[a, b, c, d]),
            Eq(self._R[x, y, z, w], expon[x, y, z, w] / sum_last_axis[x, y, z, h]),

        ], []

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []

    def init_params(self):
        pass


