from abc import abstractmethod
from joey import Layer
from joey import activation
from joey import default_name_allocator as alloc
from joey import default_dim_allocator as dim_alloc
from devito import Grid, Function, Constant, Eq, Inc, Ne, \
    ConditionalDimension
from sympy import exp, Max, And
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

        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = dim_allocator_func(10)

        gridK = Grid(shape=kernel_size, dimensions=(t1, t2, t3, t4))
        K = Function(name=name_allocator_func(), grid=gridK, space_order=0,
                     dtype=np.float64)

        gridB = Grid(shape=(input_size[0], input_size[1],
                            map_height, map_width),
                     dimensions=(t5, t6, t7, t8))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0,
                     dtype=np.float64)

        gridR = Grid(shape=(input_size[0], kernel_size[0],
                            (map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=(t5, t1, t9, t10))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        bias_grid = Grid(shape=kernel_size[0],
                         dimensions=(t1,))
        bias = Function(name=name_allocator_func(), grid=bias_grid,
                        space_order=0, dtype=np.float64)

        kernel_grad = Function(name=name_allocator_func(),
                               grid=gridK, space_order=0, dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=Grid(shape=(gridR.shape[1],
                                                gridR.shape[2],
                                                gridR.shape[3]),
                                         dimensions=(t1, t9, t10)),
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

    def equations(self):
        a, b, c, d = self._R.dimensions
        _, _, kernel_height, kernel_width = self._kernel_size
        batch_size, channels, _, _ = self._I.shape
        e, f, g, h = self._K.dimensions

        rhs = self._K[b, f, g, h] * \
            self._I[a, f, self._stride[0] * c + g,
                    self._stride[1] * d + h]

        eqs = [Inc(self._R[a, b, c, d], rhs)]

        if self._activation is not None:
            eqs.append(Eq(self._R[a, b, c, d],
                          self._activation(self._R[a, b, c, d] +
                                           self._bias[b])))
        else:
            eqs.append(Inc(self._R[a, b, c, d], self._bias[b]))

        return eqs

    def backprop_equations(self, prev_layer, next_layer, batch_constant,
                           backward_arg_dict=None):
        layer = self

        kernel_dims = layer.kernel_gradients.dimensions
        bias_dims = layer.bias_gradients.dimensions
        dims = layer.result_gradients.dimensions

        eqs = [Eq(layer.bias_gradients,
                  batch_constant * layer.bias_gradients),
               Inc(layer.bias_gradients[bias_dims[0]],
                   layer.result_gradients[dims[0], dims[1], dims[2]]),
               Eq(layer.bias_gradients,
                  layer.bias_gradients / (batch_constant + 1))]

        _, _, height, width = layer.kernel.shape

        if next_layer is not None:
            next_dims = next_layer.result_gradients.dimensions

            cd1 = ConditionalDimension(name=alloc(), parent=kernel_dims[2],
                                       condition=And(next_dims[1] - height +
                                                     1 + kernel_dims[2] >= 0,
                                                     next_dims[1] - height +
                                                     1 + kernel_dims[2] <
                                                     layer.result_gradients
                                                     .shape[1]))
            cd2 = ConditionalDimension(name=alloc(), parent=kernel_dims[3],
                                       condition=And(next_dims[2] - width + 1 +
                                                     kernel_dims[3] >= 0,
                                                     next_dims[2] - width + 1 +
                                                     kernel_dims[3] <
                                                     layer.result_gradients
                                                     .shape[2]))

            eqs += [Eq(layer.kernel_gradients,
                       batch_constant * layer.kernel_gradients),
                    Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1],
                                               kernel_dims[2], kernel_dims[3]],
                        layer.result_gradients[kernel_dims[0], dims[1],
                                               dims[2]] *
                        layer.input[batch_constant, kernel_dims[1],
                                    kernel_dims[2] + dims[1],
                                    kernel_dims[3] + dims[2]]),
                    Eq(layer.kernel_gradients,
                       layer.kernel_gradients / (batch_constant + 1)),
                    Eq(next_layer.result_gradients, 0),
                    Inc(next_layer.result_gradients[next_dims[0], next_dims[1],
                                                    next_dims[2]],
                        layer.kernel[dims[0], next_dims[0],
                                     height - kernel_dims[2] - 1,
                                     width - kernel_dims[3] - 1] *
                        layer.result_gradients[dims[0],
                                               next_dims[1] - height + 1 +
                                               kernel_dims[2],
                                               next_dims[2] - width + 1 +
                                               kernel_dims[3]],
                        implicit_dims=(cd1, cd2))] + \
                next_layer.activation.backprop_eqs(next_layer,
                                                   batch_constant)
        else:
            eqs += [Eq(layer.kernel_gradients,
                       batch_constant * layer.kernel_gradients),
                    Inc(layer.kernel_gradients[kernel_dims[0],
                                               kernel_dims[1],
                                               kernel_dims[2],
                                               kernel_dims[3]],
                        layer.result_gradients[kernel_dims[0], dims[1],
                                               dims[2]] *
                        layer.input[batch_constant, kernel_dims[1],
                                    kernel_dims[2] + dims[1],
                                    kernel_dims[3] + dims[2]]),
                    Eq(layer.kernel_gradients,
                       layer.kernel_gradients / (batch_constant + 1))]

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

        t1, t2, t3, t4, t5, t6 = dim_allocator_func(6)

        gridB = Grid(shape=(input_size[0], input_size[1], map_height,
                            map_width),
                     dimensions=(t1, t2, t3, t4))
        B = Function(name=name_allocator_func(), grid=gridB, space_order=0,
                     dtype=np.float64)

        gridR = Grid(shape=(input_size[0], input_size[1],
                            (map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=(t1, t2, t5, t6))

        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=Grid(shape=(gridR.shape[1],
                                                gridR.shape[2],
                                                gridR.shape[3]),
                                         dimensions=(t2, t5, t6)),
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
    def equations(self):
        pass

    @abstractmethod
    def backprop_equations(self, prev_layer, next_layer, batch_constant,
                           backward_arg_dict):
        pass


class MaxPooling(Pooling):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def equations(self):
        a, b, c, d = self._R.dimensions
        kernel_height, kernel_width = self._kernel_size

        rhs = Max(*[self._I[a, b,
                            self._stride[0] * c + i,
                            self._stride[1] * d + j]
                    for i in range(kernel_height)
                    for j in range(kernel_width)])

        if self._activation is not None:
            rhs = self._activation(rhs)

        return [Eq(self._R[a, b, c, d], rhs)]

    def backprop_equations(self, prev_layer, next_layer, batch_constant,
                           backward_arg_dict):
        if next_layer is None:
            return []

        layer = self

        a, b = dim_alloc(2)
        backward_arg_dict[a.name + '_M'] = layer.kernel_size[0] - 1
        backward_arg_dict[b.name + '_M'] = layer.kernel_size[1] - 1
        processed = Function(name=alloc(), grid=layer.result.grid,
                             space_order=0, dtype=np.float64)

        dims = layer.result.dimensions

        # The first dimension corresponding to a batch index must be
        # discarded here.
        dims = dims[1:]

        stride_rows, stride_cols = layer.stride

        cd1 = ConditionalDimension(name=alloc(), parent=b,
                                   condition=And(Ne(processed[batch_constant,
                                                              dims[0],
                                                              dims[1],
                                                              dims[2]], 1),
                                                 ~Ne(layer
                                                     .input[batch_constant,
                                                            dims[0],
                                                            stride_rows *
                                                            dims[1] + a,
                                                            stride_cols *
                                                            dims[2] + b],
                                                     layer.result[batch_constant,
                                                                  dims[0],
                                                                  dims[1],
                                                                  dims[2]])))

        return [Eq(next_layer.result_gradients, 0),
                Eq(processed, 0),
                Eq(next_layer.result_gradients[dims[0], stride_rows * dims[1] +
                                               a, stride_cols * dims[2] + b],
                   layer.result_gradients[dims[0], dims[1], dims[2]],
                   implicit_dims=cd1),
                Eq(processed[batch_constant, dims[0], dims[1], dims[2]],
                   1, implicit_dims=(a, b, cd1))] + \
            next_layer.activation.backprop_eqs(next_layer,
                                               batch_constant)


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

        t1, t2, t3 = dim_allocator_func(3)
        self._dimensions = (t1, t2, t3)

        gridW = Grid(shape=weight_size, dimensions=(t1, t2))
        W = Function(name=name_allocator_func(), grid=gridW, space_order=0,
                     dtype=np.float64)

        if self._input_is_vector:
            gridV_dimensions = (t2,)
            gridR_dimensions = (t1,)
            gridR_shape = weight_size[0]
            output_grad_grid = Grid(shape=gridR_shape,
                                    dimensions=gridR_dimensions)
        else:
            gridV_dimensions = (t2, t3)
            gridR_dimensions = (t1, t3)
            gridR_shape = (weight_size[0], input_size[1])
            output_grad_grid = Grid(shape=weight_size[0],
                                    dimensions=(t1,))

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
                         dimensions=(t1,))
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

    def equations(self):
        a, b, c = self._dimensions

        if self._input_is_vector:
            eqs = [Inc(self._R[a], self._K[a, b] * self._I[b])]
        else:
            eqs = [Inc(self._R[a, c], self._K[a, b] * self._I[b, c])]

        if self._activation is not None:
            eqs.append(Eq(self._R, self._activation(self._bias[a] + self._R)))
        else:
            eqs.append(Inc(self._R[a, c], self._bias[a]))

        return eqs

    def backprop_equations(self, prev_layer, next_layer, batch_constant,
                           backward_arg_dict=None):
        layer = self

        dims = layer.result_gradients.dimensions
        bias_dims = layer.bias_gradients.dimensions
        kernel_dims = layer.kernel_gradients.dimensions

        if prev_layer is None:
            return [Eq(layer.bias_gradients,
                       layer.bias_gradients * batch_constant),
                    Inc(layer.bias_gradients[bias_dims[0]],
                    layer.result_gradients[bias_dims[0]]),
                    Eq(layer.bias_gradients,
                       layer.bias_gradients / (batch_constant + 1)),
                    Eq(layer.kernel_gradients,
                       layer.kernel_gradients * batch_constant),
                    Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1]],
                        layer.input[kernel_dims[1],
                                    batch_constant] *
                        layer.result_gradients[kernel_dims[0]]),
                    Eq(layer.kernel_gradients,
                       layer.kernel_gradients / (batch_constant + 1))]

        prev_dims = prev_layer.result_gradients.dimensions

        return [Eq(layer.result_gradients, 0),
                Inc(layer.result_gradients[dims[0]],
                    prev_layer.kernel[prev_dims[0], dims[0]] *
                    prev_layer.result_gradients[prev_dims[0]])] + \
            layer.activation.backprop_eqs(layer, batch_constant) + \
            [Eq(layer.bias_gradients,
                layer.bias_gradients * batch_constant),
             Inc(layer.bias_gradients[bias_dims[0]],
                 layer.result_gradients[bias_dims[0]]),
             Eq(layer.bias_gradients,
                layer.bias_gradients / (batch_constant + 1)),
             Eq(layer.kernel_gradients,
                layer.kernel_gradients * batch_constant),
             Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1]],
                 layer.input[kernel_dims[1], batch_constant] *
                 layer.result_gradients[kernel_dims[0]]),
             Eq(layer.kernel_gradients,
                layer.kernel_gradients / (batch_constant + 1))]


class FullyConnectedSoftmax(FullyConnected):
    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, generate_code=True):
        # Size units are the same as the FullyConnected ones.

        self._name_allocator = name_allocator_func
        self._dim_allocator = dim_allocator_func
        super().__init__(weight_size, input_size, name_allocator_func,
                         dim_allocator_func, activation.Dummy(), generate_code)

    def equations(self):
        if self._input_is_vector:
            return self._equations_vector()
        else:
            return self._equations_matrix()

    def _equations_vector(self):
        C = Constant(name=self._name_allocator())
        a, b, c = self._dimensions
        return [Inc(self._T[a], self._K[a, b] * self._I[b]),
                Inc(self._T, self._bias),
                Eq(C, sum([exp(self._T[i]) for i in range(self._R.shape[0])])),
                Eq(self._R, exp(self._T) / C)]

    def _equations_matrix(self):
        a, b, c = self._dimensions

        gridC = Grid(shape=self._R.shape[1], dimensions=(c,))
        C = Function(name=self._name_allocator(), grid=gridC, space_order=0,
                     dtype=np.float64)
        M = Function(name=self._name_allocator(), grid=gridC, space_order=0,
                     dtype=np.float64)

        return [Inc(self._T[a, c], self._K[a, b] * self._I[b, c]),
                Inc(self._T[a, c], self._bias[a]),
                Eq(M[c], Max(*[self._T[i, c]
                               for i in range(self._R.shape[0])])),
                Eq(C[c], sum([exp(self._T[i, c] - M[c])
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
        t1, t2, t3, t4, t5 = dim_allocator_func(5)

        gridI = Grid(shape=input_size, dimensions=(t1, t2, t3, t4))
        I = Function(name=name_allocator_func(), grid=gridI, space_order=0,
                     dtype=np.float64)

        gridR = Grid(shape=(input_size[1]*input_size[2]*input_size[3],
                            input_size[0]),
                     dimensions=(t5, t1))
        R = Function(name=name_allocator_func(), grid=gridR, space_order=0,
                     dtype=np.float64)

        output_grad = Function(name=name_allocator_func(),
                               grid=Grid(shape=gridR.shape[0],
                                         dimensions=(t5,)),
                               space_order=0, dtype=np.float64)

        return (None, I, R, None, None, output_grad, None)

    def execute(self, input_data):
        self._I.data[:] = input_data
        return super().execute()

    def equations(self):
        _, b, c, d = self._I.dimensions
        batch_size, channels, height, width = self._I.shape

        return [Eq(self._R[b * height * width + c * height + d, a],
                   self._I[a, b, c, d]) for a in range(batch_size)]

    def backprop_equations(self, prev_layer, next_layer, batch_constant,
                           backward_arg_dict=None):
        layer = self

        prev_kernel_dims = prev_layer.kernel_gradients.dimensions
        dims = layer.result_gradients.dimensions

        _, height, width = next_layer.result_gradients.shape
        next_dims = next_layer.result_gradients.dimensions

        return [Eq(layer.result_gradients, 0),
                Inc(layer.result_gradients[dims[0]],
                    prev_layer.kernel[prev_kernel_dims[0], dims[0]] *
                    prev_layer.result_gradients[prev_kernel_dims[0]]),
                Eq(next_layer.result_gradients[next_dims[0], next_dims[1],
                                               next_dims[2]],
                   layer.result_gradients[next_dims[0] * height * width +
                                          next_dims[1] * height +
                                          next_dims[2]])] + \
            next_layer.activation.backprop_eqs(next_layer, batch_constant)
