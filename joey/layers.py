from abc import abstractmethod
from joey import Layer
from joey import activation
from joey import default_name_allocator as alloc
from joey import default_dim_allocator as dim_alloc
from devito import Grid, Function, Constant, Eq, Inc
from sympy import exp, Max
import numpy as np


# Mathematically, Conv uses a cross-correlation operation.
class Conv(Layer):
    def __init__(self, kernel_size, input_size,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=True,
                 strict_stride_check=True):
        # Kernel size argument (kernel_size) is expressed as
        # (output channels / kernel count, rows, columns).
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).
        # Input size is expressed as (batch size, channels, rows, columns).

        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        self._kernel_size = (kernel_size[0], input_size[1], kernel_size[1],
                             kernel_size[2])

        self._stride = stride
        self._padding = padding

        super().__init__(self._kernel_size, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
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

        if strict_stride_check:
            map_height = input_size[2] + 2 * padding[0]
            map_width = input_size[3] + 2 * padding[1]
            _, kernel_height, kernel_width = kernel_size

            if (map_height - kernel_height) % stride[0] != 0 or \
               (map_width - kernel_width) % stride[1] != 0:
                raise Exception("Stride " + str(stride) + " is not "
                                "compatible with feature map, kernel and "
                                "padding sizes")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        map_height = input_size[2] + 2 * self._padding[0]
        map_width = input_size[3] + 2 * self._padding[1]
        _, _, kernel_height, kernel_width = kernel_size

        gridK = Grid(shape=kernel_size, dimensions=dim_allocator_func(4))
        K = Function(name=name_allocator_func(), grid=gridK, space_order=0,
                     dtype=np.float64)

        gridB = Grid(shape=(input_size[0], input_size[1],
                            map_height, map_width),
                     dimensions=dim_allocator_func(4))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0,
                     dtype=np.float64)

        gridR = Grid(shape=(input_size[0], kernel_size[0],
                            (map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=dim_allocator_func(4))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        bias_grid = Grid(shape=kernel_size[0],
                         dimensions=dim_allocator_func(1))
        bias = Function(name=name_allocator_func(), grid=bias_grid,
                        space_order=0, dtype=np.float64)

        kernel_grad = Function(name=name_allocator_func(),
                               grid=gridK, space_order=0, dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=Grid(shape=(gridR.shape[1],
                                                gridR.shape[2],
                                                gridR.shape[3]),
                                         dimensions=dim_allocator_func(3)),
                               space_order=0, dtype=np.float64)

        bias_grad = Function(name=name_allocator_func(),
                             grid=bias_grid, space_order=0, dtype=np.float64)

        return (K, B, R, bias, kernel_grad, output_grad, bias_grad)

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

        self._R.data[:] = 0

        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        a, b, c, d = self._R.dimensions
        _, _, kernel_height, kernel_width = self._kernel_size
        batch_size, channels, _, _ = input_function.shape
        e, f, g, h = self._K.dimensions

        rhs = sum([self._K[e, f, x, y] *
                   input_function[a, f, self._stride[0] * c + x,
                                  self._stride[1] * d + y]
                   for x in range(kernel_height)
                   for y in range(kernel_width)])

        eqs = [Inc(self._R[a, e, c, d], rhs),
               Inc(self._R[a, e, c, d], self._bias[e])]

        if self._activation is not None:
            eqs.append(Eq(self._R[a, e, c, d],
                          self._activation(self._R[a, e, c, d])))

        return eqs


class Pooling(Layer):
    def __init__(self, kernel_size, input_size,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0), activation=None,
                 generate_code=True, strict_stride_check=True):
        # Kernel size is expressed as (rows, columns).
        # Input size is expressed as (batch size, channels, rows, columns).

        self._error_check(kernel_size, input_size, stride, padding,
                          strict_stride_check)

        self._kernel_size = kernel_size

        self._stride = stride
        self._padding = padding

        super().__init__(kernel_size, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code)

    def _error_check(self, kernel_size, input_size, stride, padding,
                     strict_stride_check):
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

        if strict_stride_check:
            map_height = input_size[2] + 2 * padding[0]
            map_width = input_size[3] + 2 * padding[1]
            kernel_height, kernel_width = kernel_size

            if (map_height - kernel_height) % stride[0] != 0 or \
               (map_width - kernel_width) % stride[1] != 0:
                raise Exception("Stride " + str(stride) + " is not "
                                "compatible with feature map, kernel and "
                                "padding sizes")

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        map_height = input_size[2] + 2 * self._padding[0]
        map_width = input_size[3] + 2 * self._padding[1]
        kernel_height, kernel_width = kernel_size

        a, b, c, d = dim_allocator_func(4)
        gridB = Grid(shape=(input_size[0], input_size[1], map_height,
                            map_width),
                     dimensions=(a, b, c, d))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0,
                     dtype=np.float64)

        e, f, g, h = dim_allocator_func(4)
        gridR = Grid(shape=(input_size[0], input_size[1],
                            (map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=(e, f, g, h))

        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=Grid(shape=(gridR.shape[1],
                                                gridR.shape[2],
                                                gridR.shape[3]),
                                         dimensions=dim_allocator_func(3)),
                               space_order=0, dtype=np.float64)

        return (None, B, R, None, None, output_grad, None)

    @property
    def stride(self):
        return self._stride

    @property
    def kernel_size(self):
        return self._kernel_size

    def execute(self, input_data):
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
        return super().execute()

    @abstractmethod
    def equations(self, input_function=None):
        pass


class MaxPooling(Pooling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        a, b, c, d = self._R.dimensions
        kernel_height, kernel_width = self._kernel_size

        rhs = Max(*[input_function[a, b,
                                   self._stride[0] * c + i,
                                   self._stride[1] * d + j]
                    for i in range(kernel_height)
                    for j in range(kernel_width)])

        if self._activation is not None:
            rhs = self._activation(rhs)

        return [Eq(self._R[a, b, c, d], rhs)]


class FullyConnected(Layer):
    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, activation=None,
                 generate_code=True):
        # Weight size is expressed as (rows, columns).
        # Input size is expressed as either (rows, columns) or rows.

        super().__init__(weight_size, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code)

    def _allocate(self, weight_size, input_size, name_allocator_func,
                  dim_allocator_func):
        self._input_is_vector = type(input_size) == int

        self._dimensions = dim_allocator_func(3)
        a, b, c = self._dimensions

        gridW = Grid(shape=weight_size, dimensions=(a, b))
        W = Function(name=name_allocator_func(), grid=gridW, space_order=0,
                     dtype=np.float64)

        if self._input_is_vector:
            gridV_dimensions = (b,)
            gridR_dimensions = (a,)
            gridR_shape = weight_size[0]
            output_grad_grid = Grid(shape=gridR_shape,
                                    dimensions=dim_allocator_func(1))
        else:
            gridV_dimensions = (b, c)
            gridR_dimensions = (a, c)
            gridR_shape = (weight_size[0], input_size[1])
            output_grad_grid = Grid(shape=weight_size[0],
                                    dimensions=dim_allocator_func(1))

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
                         dimensions=dim_allocator_func(1))
        bias = Function(name=name_allocator_func(), grid=bias_grid,
                        space_order=0, dtype=np.float64)

        kernel_grad = Function(name=name_allocator_func(),
                               grid=gridW, space_order=0, dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=output_grad_grid, space_order=0,
                               dtype=np.float64)

        bias_grad = Function(name=name_allocator_func(),
                             grid=bias_grid, space_order=0, dtype=np.float64)

        return (W, V, R, bias, kernel_grad, output_grad, bias_grad)

    def execute(self, input_data, bias, weight_data=None):
        if weight_data is not None:
            self._K.data[:] = weight_data

        self._I.data[:] = input_data
        self._bias.data[:] = bias

        if self._activation is not None:
            self._T.data[:] = 0

        self._R.data[:] = 0

        return super().execute()

    def equations(self, input_function=None):
        if input_function is None:
            input_function = self._I

        a, b, c = self._dimensions

        if self._input_is_vector:
            eqs = [Inc(self._R[a], self._K[a, b] * input_function[b])]
        else:
            eqs = [Inc(self._R[a, c], self._K[a, b] * input_function[b, c])]

        eqs.append(Inc(self._R[a, c], self._bias[a]))

        if self._activation is not None:
            eqs.append(Eq(self._R, self._activation(self._R)))

        return eqs


class FullyConnectedSoftmax(FullyConnected):
    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, generate_code=True):
        # Size units are the same as the FullyConnected ones.

        self._name_allocator = name_allocator_func
        self._dim_allocator = dim_allocator_func
        super().__init__(weight_size, input_size, name_allocator_func,
                         dim_allocator_func, activation.Dummy(), generate_code)

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
        C = Function(name=self._name_allocator(), grid=gridC, space_order=0,
                     dtype=np.float64)
        M = Function(name=self._name_allocator(), grid=gridC, space_order=0,
                     dtype=np.float64)
        x = C.dimensions[0]
        a, b, c = self._dimensions

        return [Inc(self._T[a, c], self._K[a, b] * input_function[b, c]),
                Inc(self._T[a, c], self._bias[a]),
                Eq(M[x], Max(*[self._T[i, x]
                               for i in range(self._R.shape[0])])),
                Eq(C[x], sum([exp(self._T[i, x] - M[x])
                              for i in range(self._R.shape[0])])),
                Eq(self._R[a, b], exp(self._T[a, b] - M[b]) / C[b]),
                Eq(self._T, 0)]


class Flat(Layer):
    def __init__(self, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, generate_code=True):
        # Input size is expressed as (batch size, channels, rows, columns).

        super().__init__(None, input_size, None, name_allocator_func,
                         dim_allocator_func, generate_code)

    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func):
        gridI = Grid(shape=input_size, dimensions=dim_allocator_func(4))
        I = Function(name=name_allocator_func(), grid=gridI, space_order=0,
                     dtype=np.float64)

        gridR = Grid(shape=(input_size[1]*input_size[2]*input_size[3],
                            input_size[0]),
                     dimensions=dim_allocator_func(2))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=Grid(shape=gridR.shape[0],
                                         dimensions=dim_allocator_func(1)),
                               space_order=0, dtype=np.float64)

        return (None, I, R, None, None, output_grad, None)

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
