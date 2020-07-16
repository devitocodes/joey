from devito.ml import Layer
from devito.ml import default_name_allocator as alloc
from devito.ml import default_dim_allocator as dim_alloc
from devito import Grid, Function, Constant, Eq, Inc
from sympy import exp
import numpy as np


# Mathematically, ConvConv uses a convolution operation.
class ConvConv(Layer):
    def __init__(self, kernel_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, activation=None,
                 generate_code=True):
        # All sizes are expressed as (rows, columns).
        # No batches/multiple channels are supported.

        self._activation = activation
        self._bias = Constant(name=name_allocator_func())

        super().__init__(kernel_size, input_size, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _error_check(self, kernel_size, input_size):
        if kernel_size is None or len(kernel_size) != 2:
            raise Exception("kernel_size is incorrect")

        if input_size is None or len(input_size) != 2:
            raise Exception("input_size is incorrect")

        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise Exception("The dimensions of kernel must be odd")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        self._error_check(kernel_size, input_size)

        gridA = Grid(shape=kernel_size,
                     dimensions=dim_allocator_func(2))
        A = Function(name=name_allocator_func(), grid=gridA, space_order=0)

        gridBR = Grid(shape=input_size)

        B = Function(name=name_allocator_func(), grid=gridBR, space_order=1)
        R = Function(name=name_allocator_func(), grid=gridBR, space_order=0)

        return (A, B, R)

    def execute(self, input_data, bias, kernel_data=None):
        if kernel_data is not None:
            self._K.data[:] = kernel_data

        self._I.data[:] = input_data
        self._bias.data = bias

        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        x, y = input_function.dimensions
        kernel_rows, kernel_cols = self._K.shape

        rhs = sum([self._K[kernel_rows - i - 1,
                           kernel_cols - j - 1] *
                   input_function[x - kernel_rows // 2 + i,
                                  y - kernel_cols // 2 + j]
                   for i in range(kernel_rows)
                   for j in range(kernel_cols)]) + self._bias

        if self._activation is not None:
            rhs = self._activation(rhs)

        return [Eq(self._R[x, y], rhs)]


# Mathematically, Conv uses a cross-correlation operation.
class Conv(Layer):
    def __init__(self, kernel_size, input_size,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=True):
        # Kernel size is expressed as (kernel count, rows, columns).
        # Input size is expressed as (batch size, channels, rows, columns).

        self._error_check(kernel_size, input_size, stride, padding)

        self._kernel_size = kernel_size
        self._activation = activation
        self._bias = Function(name=name_allocator_func(),
                              grid=Grid(shape=kernel_size[0],
                                        dimensions=dim_allocator_func(1)),
                              space_order=0)

        self._stride = stride
        self._padding = padding

        super().__init__(kernel_size, input_size, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding):
        if input_size is None or len(input_size) != 4:
            raise Exception("Input size is incorrect")

        if kernel_size is None or len(kernel_size) != 3:
            raise Exception("Kernel size is incorrect")

        if stride is None or len(stride) != 2:
            raise Exception("Stride is incorrect")

        if stride[0] < 1 or stride[1] < 1:
            raise Exception("Stride cannot be less than 1")

        if padding is None or len(padding) != 2:
            raise Exception("Padding is incorrect")

        if padding[0] < 0 or padding[1] < 0:
            raise Exception("Padding cannot be negative")

        map_height = input_size[2] + 2 * padding[0]
        map_width = input_size[3] + 2 * padding[1]
        kernel_count, kernel_height, kernel_width = kernel_size

        if (map_height - kernel_height) % stride[0] != 0 or \
           (map_width - kernel_width) % stride[1] != 0:
            raise Exception("Stride " + str(stride) + " is not "
                            "compatible with feature map, kernel and padding "
                            "sizes")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        map_height = input_size[2] + 2 * self._padding[0]
        map_width = input_size[3] + 2 * self._padding[1]
        kernel_count, kernel_height, kernel_width = kernel_size

        gridK = Grid(shape=kernel_size, dimensions=dim_allocator_func(3))
        K = Function(name=name_allocator_func(), grid=gridK, space_order=0)

        gridB = Grid(shape=(input_size[0], input_size[1],
                            map_height, map_width),
                     dimensions=dim_allocator_func(4))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0)

        gridR = Grid(shape=(input_size[0], kernel_size[0],
                            (map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=dim_allocator_func(4))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0)

        return (K, B, R)

    def execute(self, input_data, bias, kernel_data=None):
        map_height = input_data.shape[2] + 2 * self._padding[0]
        batch_size, channels, _, _ = input_data.shape

        for i in range(batch_size):
            for j in range(channels):
                for k in range(self._padding[0],
                               map_height - self._padding[0]):
                    self._I.data[i, j, k] = \
                        np.concatenate(([0] * self._padding[1],
                                        input_data[i, j, k - self._padding[0]],
                                        [0] * self._padding[1]))

        if kernel_data is not None:
            self._K.data[:] = kernel_data

        self._bias.data[:] = bias

        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        a, b, c, d = input_function.dimensions
        kernel_count, kernel_height, kernel_width = self._kernel_size
        batch_size, channels, _, _ = input_function.shape
        e, f, g = self._K.dimensions

        eqs = []

        for i in range(batch_size):
            rhs = sum([self._K[e, x, y] *
                       input_function[i, z, self._stride[0] * c + x,
                                      self._stride[1] * d + y]
                       for z in range(channels)
                       for x in range(kernel_height)
                       for y in range(kernel_width)]) + self._bias[e]

            if self._activation is not None:
                rhs = self._activation(rhs)

            eqs.append(Eq(self._R[i, e, c, d], rhs))

        return eqs


class Subsampling(Layer):
    def __init__(self, kernel_size, input_size, function,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0), activation=None,
                 generate_code=True):
        # Kernel size is expressed as (rows, columns).
        # Input size is expressed as (batch size, channels, rows, columns).

        self._error_check(kernel_size, input_size, stride, padding)

        self._kernel_size = kernel_size
        self._function = function
        self._activation = activation
        self._bias = Function(name=name_allocator_func(),
                              grid=Grid(shape=input_size[1],
                                        dimensions=dim_allocator_func(1)),
                              space_order=0)

        self._stride = stride
        self._padding = padding

        super().__init__(kernel_size, input_size, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding):
        if input_size is None or len(input_size) != 4:
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

        map_height = input_size[2] + 2 * padding[0]
        map_width = input_size[3] + 2 * padding[1]
        kernel_height, kernel_width = kernel_size

        if (map_height - kernel_height) % stride[0] != 0 or \
           (map_width - kernel_width) % stride[1] != 0:
            raise Exception("Stride " + str(stride) + " is not "
                            "compatible with feature map, kernel and padding "
                            "sizes")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        map_height = input_size[2] + 2 * self._padding[0]
        map_width = input_size[3] + 2 * self._padding[1]
        kernel_height, kernel_width = kernel_size

        a, b, c, d = dim_allocator_func(4)
        gridB = Grid(shape=(input_size[0], input_size[1], map_height,
                            map_width),
                     dimensions=(a, b, c, d))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0)

        e, f, g, h = dim_allocator_func(4)
        gridR = Grid(shape=(input_size[0], input_size[1],
                            (map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=(e, f, g, h))

        R = Function(name=name_allocator_func(), grid=gridR, space_order=0)
        return (None, B, R)

    def execute(self, input_data, bias):
        map_height = input_data.shape[2]

        # Add padding to the start and end of each row
        for image in range(input_data.shape[0]):
            for channel in range(input_data.shape[1]):
                for i in range(self._padding[0],
                               map_height - self._padding[0]):
                    self._I.data[image, channel, i] = \
                        np.concatenate(([0] * self._padding[1],
                                        input_data[image, channel,
                                                   i - self._padding[0]],
                                        [0] * self._padding[1]))

        self._bias.data[:] = bias
        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        a, b, c, d = input_function.dimensions
        kernel_height, kernel_width = self._kernel_size
        images = input_function.shape[0]
        channels = input_function.shape[1]
        equation_sum = []
        for image in range(images):
            for channel in range(channels):
                rhs = self._function([input_function[image, channel,
                                                     self._stride[0] * c + i,
                                                     self._stride[1] * d + j]
                                      for i in range(kernel_height)
                                      for j in range(kernel_width)]) \
                    + self._bias[channel]

                if self._activation is not None:
                    rhs = self._activation(rhs)

                equation_sum.append(Eq(self._R[image, channel, c, d], rhs))

        return equation_sum


class FullyConnected(Layer):
    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, activation=None,
                 generate_code=True):
        # Weight size is expressed as (rows, columns).
        # Input size is expressed as either (rows, columns) or rows.

        self._activation = activation
        self._bias = Constant(name=name_allocator_func())

        super().__init__(weight_size, input_size, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _allocate(self, weight_size, input_size, name_allocator_func,
                  dim_allocator_func):
        self._input_is_vector = type(input_size) == int

        self._dimensions = dim_allocator_func(3)
        a, b, c = self._dimensions

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

    def execute(self, input_data, bias, weight_data=None):
        if weight_data is not None:
            self._K.data[:] = weight_data

        self._I.data[:] = input_data
        self._bias.data = bias

        if self._activation is not None:
            self._T.data[:] = 0

        self._R.data[:] = 0

        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        a, b, c = self._dimensions

        if self._activation is not None:
            if self._input_is_vector:
                eqs = [Inc(self._T[a], self._K[a, b] * input_function[b])]
            else:
                eqs = [Inc(self._T[a, c],
                           self._K[a, b] * input_function[b, c])]

            eqs.append(Eq(self._R, self._activation(self._T + self._bias)))
            return eqs

        if self._input_is_vector:
            eqs = [Inc(self._R[a], self._K[a, b] * input_function[b])]
        else:
            eqs = [Inc(self._R[a, c], self._K[a, b] * input_function[b, c])]

        eqs.append(Inc(self._R, self._bias))
        return eqs


class FullyConnectedSoftmax(FullyConnected):
    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, generate_code=True):
        # Size units are the same as the FullyConnected ones.

        self._name_allocator = name_allocator_func
        self._dim_allocator = dim_allocator_func
        super().__init__(weight_size, input_size, name_allocator_func,
                         dim_allocator_func, lambda a: None, generate_code)

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        if self._input_is_vector:
            return self._equations_vector(input_function)
        else:
            return self._equations_matrix(input_function)

    def _equations_vector(self, input_function):
        C = Constant(name=self._name_allocator())
        a, b, c = self._dimensions
        return [Inc(self._T[a], self._K[a, b] * input_function[b]),
                Inc(self._T, self._bias),
                Eq(C, sum([exp(self._T[i]) for i in range(self._R.shape[0])])),
                Eq(self._R, exp(self._T) / C)]

    def _equations_matrix(self, input_function):
        gridC = Grid(shape=self._R.shape[1], dimensions=self._dim_allocator(1))
        C = Function(name=self._name_allocator(), grid=gridC, space_order=0)
        x = C.dimensions[0]
        a, b, c = self._dimensions

        return [Inc(self._T[a, c], self._K[a, b] * input_function[b, c]),
                Inc(self._T, self._bias),
                Eq(C[x], sum([exp(self._T[i, x])
                              for i in range(self._R.shape[0])])),
                Eq(self._R[a, b], exp(self._T[a, b]) / C[b])]


class Flat(Layer):
    def __init__(self, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, generate_code=True):
        # Input size is expressed as (batch size, channels, rows, columns).

        super().__init__(None, input_size, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        gridI = Grid(shape=input_size, dimensions=dim_allocator_func(4))
        I = Function(name=name_allocator_func(), grid=gridI, space_order=0)

        gridR = Grid(shape=(input_size[1]*input_size[2]*input_size[3],
                            input_size[0]),
                     dimensions=dim_allocator_func(2))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0)

        return (None, I, R)

    def execute(self, input_data):
        self._I.data[:] = input_data
        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        _, b, c, d = input_function.dimensions
        batch_size, channels, height, width = input_function.shape

        return [Eq(self._R[b * height * width + c * height + d, a],
                   input_function[a, b, c, d]) for a in range(batch_size)]
