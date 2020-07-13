from devito.ml import Layer
from devito.ml import default_name_allocator as alloc
from devito import Grid, Function, Constant, dimensions, Eq, Inc
from sympy import exp
import numpy as np


# Mathematically, ConvConv uses a convolution operation.
class ConvConv(Layer):
    def __init__(self, kernel_size, input_size, name_allocator_func=alloc,
                 activation=None, generate_code=True):
        self._activation = activation
        self._bias = Constant(name=name_allocator_func())

        super().__init__(kernel_size, input_size, name_allocator_func,
                         generate_code)

    def _error_check(self, kernel_size, input_size):
        if kernel_size is None or len(kernel_size) != 2:
            raise Exception("kernel_size is incorrect")

        if input_size is None or len(input_size) != 2:
            raise Exception("input_size is incorrect")

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise Exception("The dimensions of kernel must be odd")

    def _allocate(self, kernel_size, input_size, name_allocator_func):
        self._error_check(kernel_size, input_size)

        gridA = Grid(shape=kernel_size,
                     dimensions=dimensions('m n'))
        A = Function(name=name_allocator_func(), grid=gridA, space_order=0)

        gridBR = Grid(shape=input_size)

        B = Function(name=name_allocator_func(), grid=gridBR, space_order=1)
        R = Function(name=name_allocator_func(), grid=gridBR, space_order=0)

        return (A, B, R)

    def execute(self, kernel_data, input_data, bias):
        self._K.data[:] = kernel_data
        self._I.data[:] = input_data
        self._bias.data = bias

        return super().execute()

    def equations(self):
        x, y = self._I.dimensions
        kernel_rows, kernel_cols = self._K.shape

        rhs = sum([self._K[kernel_rows - i - 1,
                           kernel_cols - j - 1] *
                   self._I[x - kernel_rows // 2 + i,
                           y - kernel_cols // 2 + j]
                   for i in range(kernel_rows)
                   for j in range(kernel_cols)]) + self._bias

        if self._activation is not None:
            rhs = self._activation(rhs)

        return [Eq(self._R[x, y], rhs)]


# Mathematically, Conv uses a cross-correlation operation.
class Conv(Layer):
    def __init__(self, kernel_size, input_size, name_allocator_func=alloc,
                 stride=(1, 1), padding=(0, 0), activation=None,
                 generate_code=True):
        self._error_check(self._kernel, input_size, self._stride,
                          self._padding)

        self._stride = stride
        self._padding = padding
        self._kernel_size = kernel_size

        self._layer = Subsampling(kernel_size=kernel_size,
                                  input_size=input_size,
                                  function=self._convolve,
                                  name_allocator_func=name_allocator_func,
                                  stride=stride,
                                  padding=padding,
                                  activation=activation,
                                  generate_code=generate_code)

        super().__init__(kernel_size, input_size, name_allocator_func,
                         generate_code)

    def _convolve(self, values):
        acc = 0

        for i in range(self._kernel_size[0]):
            for j in range(self._kernel_size[1]):
                acc += self._kernel[i][j] * values[i * self._kernel_size[0] + j]

        return acc

    def _error_check(self, kernel_size, input_size, stride, padding):
        if input_size is None or len(input_size) != 2:
            raise Exception("Input size is incorrect")

        if kernel_size is None or len(kernel_size) != 2:
            raise Exception("Kernel size is incorrect")

        if stride is None or len(stride) != 2:
            raise Exception("Stride is incorrect")

        if stride[0] < 1 or stride[1] < 1:
            raise Exception("Stride cannot be less than 1")

        if padding is None or len(padding) != 2:
            raise Exception("Padding is incorrect")

        if padding[0] < 0 or padding[1] < 0:
            raise Exception("Padding cannot be negative")

        map_height = input_size[0] + 2 * padding[0]
        map_width = input_size[1] + 2 * padding[1]
        kernel_height, kernel_width = kernel_size

        if (map_height - kernel_height) % stride[0] != 0 or \
           (map_width - kernel_width) % stride[1] != 0:
            raise Exception("Stride " + str(stride) + " is not "
                            "compatible with input data, kernel and padding "
                            "sizes")

    def _allocate(self, kernel_size, input_size, name_allocator_func):
        pass

    def execute(self, kernel_data, input_data, bias):
        return self._layer.execute(kernel_data, input_data, bias)

    def equations(self):
        return self._layer.equations()


class Subsampling(Layer):
    def __init__(self, kernel_size, input_size, function,
                 name_allocator_func=alloc, stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=True):
        # All sizes are expressed as (rows, columns).
        self._error_check(kernel_size, input_size, stride, padding)

        self._kernel_size = kernel_size
        self._function = function
        self._activation = activation
        self._bias = Constant(name=name_allocator_func())

        self._stride = stride
        self._padding = padding

        super().__init__(kernel_size, input_size, name_allocator_func,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding):
        if input_size is None or len(input_size) != 2:
            raise Exception("Input size is incorrect")

        if kernel_size is None or len(kernel_size) != 2:
            raise Exception("Kernel size is incorrect")

        if stride is None or len(stride) != 2:
            raise Exception("Stride is incorrect")

        if stride[0] < 1 or stride[1] < 1:
            raise Exception("Stride cannot be less than 1")

        if padding is None or len(padding) != 2:
            raise Exception("Padding is incorrect")

        if padding[0] < 0 or padding[1] < 0:
            raise Exception("Padding cannot be negative")

        map_height = input_size[0] + 2 * padding[0]
        map_width = input_size[1] + 2 * padding[1]
        kernel_height, kernel_width = kernel_size

        if (map_height - kernel_height) % stride[0] != 0 or \
           (map_width - kernel_width) % stride[1] != 0:
            raise Exception("Stride " + str(stride) + " is not "
                            "compatible with feature map, kernel and padding "
                            "sizes")

    def _allocate(self, kernel_size, input_size, name_allocator_func):
        map_height = input_size[0] + 2 * self._padding[0]
        map_width = input_size[1] + 2 * self._padding[1]
        kernel_height, kernel_width = kernel_size

        gridB = Grid(shape=(map_height, map_width))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0)

        a, b = dimensions('a b')
        gridR = Grid(shape=((map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=(a, b))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0)

        return (None, B, R)

    def execute(self, input_data, bias):
        map_height = len(input_data) + 2 * self._padding[0]

        for i in range(self._padding[0], map_height - self._padding[0]):
            self._I.data[i] = \
                np.concatenate(([0] * self._padding[1],
                                input_data[i - self._padding[0]],
                                [0] * self._padding[1]))
        self._bias.data = bias

        return super().execute()

    def equations(self):
        a, b = self._I.dimensions
        kernel_height, kernel_width = self._kernel_size

        rhs = self._function([self._I[self._stride[0] * a + i,
                                      self._stride[1] * b + j]
                              for i in range(kernel_height)
                              for j in range(kernel_width)]) + self._bias

        if self._activation is not None:
            rhs = self._activation(rhs)

        return [Eq(self._R[a, b], rhs)]


class FullyConnected(Layer):
    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 activation=None, generate_code=True):
        self._activation = activation
        self._bias = Constant(name=name_allocator_func())

        super().__init__(weight_size, input_size, name_allocator_func,
                         generate_code)

    def _allocate(self, weight_size, input_size, name_allocator_func):
        self._input_is_vector = type(input_size) == int

        a, b, c = dimensions('a b c')

        gridW = Grid(shape=weight_size, dimensions=(a, b))
        W = Function(name=name_allocator_func(), grid=gridW, space_order=0)

        if self._input_is_vector:
            gridV_dimensions = (b,)
            gridR_dimensions = (a,)
            gridR_shape = weight_size[0]
        else:
            gridV_dimensions = (b, c)
            gridR_dimensions = (a, c)
            gridR_shape = (weight_size[0], input_size[1])

        gridV = Grid(shape=input_size, dimensions=gridV_dimensions)
        V = Function(name=name_allocator_func(), grid=gridV, space_order=0)

        gridR = Grid(shape=gridR_shape, dimensions=gridR_dimensions)
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0)

        if self._activation is not None:
            self._T = Function(name=name_allocator_func(), grid=gridR,
                               space_order=0)

        return (W, V, R)

    def execute(self, weight_data, input_data, bias):
        self._K.data[:] = weight_data
        self._I.data[:] = input_data
        self._bias.data = bias

        if self._activation is not None:
            self._T.data[:] = 0

        self._R.data[:] = 0

        return super().execute()

    def equations(self):
        if self._activation is not None:
            return [Inc(self._T, self._K * self._I),
                    Eq(self._R, self._activation(self._T + self._bias))]

        return [Inc(self._R, self._K * self._I), Inc(self._R, self._bias)]


class FullyConnectedSoftmax(FullyConnected):
    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 generate_code=True):
        self._name_allocator = name_allocator_func
        super().__init__(weight_size, input_size, name_allocator_func,
                         lambda a: None, generate_code)

    def equations(self):
        if self._input_is_vector:
            return self._equations_vector()
        else:
            return self._equations_matrix()

    def _equations_vector(self):
        C = Constant(name=self._name_allocator())
        return [Inc(self._T, self._K * self._I),
                Inc(self._T, self._bias),
                Eq(C, sum([exp(self._T[i]) for i in range(self._R.shape[0])])),
                Eq(self._R, exp(self._T) / C)]

    def _equations_matrix(self):
        gridC = Grid(shape=self._R.shape[1])
        C = Function(name=self._name_allocator(), grid=gridC, space_order=0)
        x = C.dimensions[0]
        a, b = self._R.dimensions

        return [Inc(self._T, self._K * self._I),
                Inc(self._T, self._bias),
                Eq(C[x], sum([exp(self._T[i, x])
                              for i in range(self._R.shape[0])])),
                Eq(self._R[a, b], exp(self._T[a, b]) / C[b])]
