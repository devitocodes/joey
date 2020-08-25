from abc import abstractmethod
from joey import Layer
from joey import activation
from joey import default_name_allocator as alloc
from joey import default_dim_allocator as dim_alloc
from devito import Grid, Function, Constant, Eq, Inc, \
    ConditionalDimension
from sympy import exp, Max, And, Min, sign
import numpy as np


class Conv(Layer):
    """
    A Layer subclass corresponding to a 2D convolution layer (mathematically,
    it performs a cross-correlation operation).

    Parameters
    ----------
    kernel_size : (int, int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (output channels / kernel count, rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0),
                 activation=None, generate_code=False,
                 strict_stride_check=True):
        # Internal kernel size (self._kernel_size) is expressed as
        # (output channels / kernel count, input channels, rows, columns).

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
                                "padding sizes. If you want to proceed "
                                "anyway, set strict_stride_check=False when "
                                "instantiating this object")

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
                               grid=gridR,
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

        return (eqs, [])

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        kernel_dims = layer.kernel_gradients.dimensions
        bias_dims = layer.bias_gradients.dimensions
        dims = layer.result_gradients.dimensions

        eqs = [Inc(layer.bias_gradients[bias_dims[0]],
                   layer.result_gradients[dims[0], dims[1], dims[2], dims[3]]),
               Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1],
                                          kernel_dims[2], kernel_dims[3]],
                   layer.result_gradients[dims[0],
                                          kernel_dims[0], dims[2],
                                          dims[3]] *
                   layer.input[dims[0], kernel_dims[1],
                               kernel_dims[2] + dims[2],
                               kernel_dims[3] + dims[3]])]

        _, _, height, width = layer.kernel.shape

        if next_layer is not None:
            next_dims = next_layer.result_gradients.dimensions

            cd1 = ConditionalDimension(name=alloc(), parent=kernel_dims[2],
                                       condition=And(next_dims[2] - height +
                                                     1 + kernel_dims[2] >= 0,
                                                     next_dims[2] - height +
                                                     1 + kernel_dims[2] <
                                                     layer.result_gradients
                                                     .shape[2]))
            cd2 = ConditionalDimension(name=alloc(), parent=kernel_dims[3],
                                       condition=And(next_dims[3] - width + 1 +
                                                     kernel_dims[3] >= 0,
                                                     next_dims[3] - width + 1 +
                                                     kernel_dims[3] <
                                                     layer.result_gradients
                                                     .shape[3]))

            eqs += [Inc(next_layer.result_gradients[next_dims[0],
                                                    next_dims[1],
                                                    next_dims[2],
                                                    next_dims[3]],
                        layer.kernel[dims[1], next_dims[1],
                                     height - kernel_dims[2] - 1,
                                     width - kernel_dims[3] - 1] *
                        layer.result_gradients[next_dims[0],
                                               dims[1],
                                               next_dims[2] - height + 1 +
                                               kernel_dims[2],
                                               next_dims[3] - width + 1 +
                                               kernel_dims[3]],
                        implicit_dims=(cd1, cd2))] + \
                next_layer.activation.backprop_eqs(next_layer)

        return (eqs, [])


class Pooling(Layer):
    """
    A Layer abstract subclass corresponding to a generic pooling layer.
    When you create a subclass of Pooling, you have to implement
    the following methods: equations(), backprop_equations().

    Parameters
    ----------
    kernel_size : (int, int)
        The shape of a kernel (represented internally by a NumPy array)
        expressed as (rows, columns).
    input_size : (int, int, int, int)
        The shape of input data expressed as
        (batch size, channels, rows, columns).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    stride : (int, int), optional
        Stride of the layer expressed as (rows, columns). The default
        value is (1, 1).
    padding : (int, int), optional
        Padding of the layer expressed as (rows, columns). The default
        value is (0, 0).

        Be careful! The current version of Joey supports non-zero padding
        ONLY for standalone layers. When you create a neural network, all
        of its layers must have (0, 0) padding.
    activation : Activation, optional
        See Layer.__doc__. The actual default value is Dummy.
    generate_code : bool, optional
        See Layer.__doc__.
    strict_stride_check : bool, optional
        A boolean indicating whether a strict stride check should be
        performed when instantiating this object. The default value is
        True.

        If the check is disabled and the stride turns out to be
        incompatible with the provided kernel, input and padding sizes,
        some parts of input data will not be processed. This behaviour
        is intentional, its aim is avoiding any out-of-bounds accesses.
    """

    def __init__(self, kernel_size, input_size,
                 name_allocator_func=alloc, dim_allocator_func=dim_alloc,
                 stride=(1, 1), padding=(0, 0), activation=None,
                 generate_code=False, strict_stride_check=True):
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
                                "padding sizes. If you want to proceed "
                                "anyway, set strict_stride_check=False "
                                "when instantiating this object")

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
                               grid=gridR,
                               space_order=0, dtype=np.float64)

        return (None, B, R, None, None, output_grad, None)

    @property
    def stride(self):
        """Stride of the layer."""
        return self._stride

    @property
    def kernel_size(self):
        """The kernel size of the layer."""
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
    def backprop_equations(self, prev_layer, next_layer):
        pass


class MaxPooling(Pooling):
    """
    A Layer/Pooling subclass corresponding to a max pooling layer.

    Parameters
    ----------
    See Pooling.__doc__.
    """

    def __init__(self, *args, **kwargs):
        self._indices = None
        self._forward_tmp_constants = None
        self._backward_tmp_constants = None
        super().__init__(*args, **kwargs)

    def equations(self):
        if self._forward_tmp_constants is None:
            self._forward_tmp_constants = \
                [Constant(name=alloc(), dtype=np.float64)]

        if self._indices is None:
            self._indices = \
                Function(name=alloc(),
                         grid=self._R.grid,
                         space_order=0,
                         dtype=np.int32)

        a, b, c, d = self._R.dimensions
        kernel_height, kernel_width = self._kernel_size
        i, j = dim_alloc(2)

        args = [(i.name + '_M', kernel_height - 1),
                (j.name + '_M', kernel_width - 1)]

        old = self._forward_tmp_constants[0]

        cond1 = abs(sign(self._R[a, b, c, d] - old)) * kernel_width * \
            kernel_height
        cond2 = abs(sign(self._I[a, b, self._stride[0] * c + i,
                                 self._stride[1] * d + j] -
                         self._R[a, b, c, d])) * kernel_width * kernel_height

        eqs = [Eq(self._indices, kernel_height * kernel_width),
               Eq(self._R[a, b, c, d], self._I[a, b,
                                               self._stride[0] * c,
                                               self._stride[1] * d]),
               Eq(old, self._R[a, b, c, d], implicit_dims=(i, j)),
               Eq(self._R[a, b, c, d], Max(self._R[a, b, c, d],
                                           self._I[a, b,
                                                   self._stride[0] * c + i,
                                                   self._stride[1] * d + j])),
               Eq(self._indices[a, b, c, d],
                  Min(self._indices[a, b, c, d] + cond1,
                      i * kernel_width + j + cond2))]

        if self._activation is not None:
            eqs.append(Eq(self._R, self._activation(self._R)))

        return (eqs, args)

    def backprop_equations(self, prev_layer, next_layer):
        if next_layer is None:
            return ([], [])

        if self._backward_tmp_constants is None:
            self._backward_tmp_constants = \
                [Constant(name=alloc(), dtype=np.int32),
                 Constant(name=alloc(), dtype=np.int32)]

        dims = self._R.dimensions
        stride_rows, stride_cols = self.stride

        index = self._indices[dims[0], dims[1], dims[2], dims[3]]
        a = self._backward_tmp_constants[0]
        b = self._backward_tmp_constants[1]

        return ([Eq(a, index // 2),
                 Eq(b, index % 2),
                 Inc(next_layer.result_gradients[dims[0],
                                                 dims[1],
                                                 stride_rows * dims[2] + a,
                                                 stride_cols * dims[3] + b],
                     self.result_gradients[dims[0],
                                           dims[1], dims[2], dims[3]])] +
                next_layer.activation.backprop_eqs(next_layer), [])


class FullyConnected(Layer):
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

    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, activation=None,
                 generate_code=False):
        super().__init__(weight_size, input_size, activation,
                         name_allocator_func, dim_allocator_func,
                         generate_code)

    def _allocate(self, weight_size, input_size, name_allocator_func,
                  dim_allocator_func):
        t1, t2, t3 = dim_allocator_func(3)
        self._dimensions = (t1, t2, t3)

        gridW = Grid(shape=weight_size, dimensions=(t1, t2))
        W = Function(name=name_allocator_func(), grid=gridW, space_order=0,
                     dtype=np.float64)

        gridV_dimensions = (t2, t3)
        gridR_dimensions = (t1, t3)
        gridR_shape = (weight_size[0], input_size[1])

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
                               grid=gridR, space_order=0,
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

        eqs = [Inc(self._R[a, c], self._K[a, b] * self._I[b, c])]

        if self._activation is not None:
            eqs.append(Eq(self._R, self._activation(self._bias[a] + self._R)))
        else:
            eqs.append(Inc(self._R[a, c], self._bias[a]))

        return (eqs, [])

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        dims = layer.result_gradients.dimensions
        kernel_dims = layer.kernel_gradients.dimensions

        if prev_layer is None:
            return ([Inc(layer.bias_gradients, layer.result_gradients),
                     Inc(layer.kernel_gradients[kernel_dims[0],
                                                kernel_dims[1]],
                         layer.input[kernel_dims[1],
                                     dims[1]] *
                         layer.result_gradients[kernel_dims[0], dims[1]])], [])

        prev_dims = prev_layer.result_gradients.dimensions

        return ([Inc(layer.result_gradients[dims[0], dims[1]],
                     prev_layer.kernel[prev_dims[0], dims[0]] *
                     prev_layer.result_gradients[prev_dims[0], dims[1]])] +
                layer.activation.backprop_eqs(layer) +
                [Inc(layer.bias_gradients, layer.result_gradients),
                 Eq(layer.kernel_gradients[kernel_dims[0], kernel_dims[1]],
                    layer.kernel_gradients[kernel_dims[0], kernel_dims[1]] +
                    layer.input[kernel_dims[1], dims[1]] *
                    layer.result_gradients[kernel_dims[0], dims[1]])], [])


class FullyConnectedSoftmax(FullyConnected):
    """
    A Layer/FullyConnected subclass corresponding to a full connection (FC)
    layer with the softmax activation.

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
    generate_code : bool, optional
        See Layer.__doc__.
    """

    def __init__(self, weight_size, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, generate_code=False):
        self._name_allocator = name_allocator_func
        self._dim_allocator = dim_allocator_func
        super().__init__(weight_size, input_size, name_allocator_func,
                         dim_allocator_func, activation.Dummy(), generate_code)

    def equations(self):
        a, b, c = self._dimensions

        gridC = Grid(shape=self._R.shape[1], dimensions=(c,))
        C = Function(name=self._name_allocator(), grid=gridC, space_order=0,
                     dtype=np.float64)
        M = Function(name=self._name_allocator(), grid=gridC, space_order=0,
                     dtype=np.float64)

        return ([Inc(self._T[a, c], self._K[a, b] * self._I[b, c]),
                 Inc(self._T[a, c], self._bias[a]),
                 Eq(M[c], Max(*[self._T[i, c]
                                for i in range(self._R.shape[0])])),
                 Eq(C[c], sum([exp(self._T[i, c] - M[c])
                               for i in range(self._R.shape[0])])),
                 Eq(self._R[a, b], exp(self._T[a, b] - M[b]) / C[b]),
                 Eq(self._T, 0)], [])


class Flat(Layer):
    """
    A Layer subclass corresponding to an internal flattening layer turning
    a 4D array into a 2D matrix required by a full connection (FC) layer.

    When creating a neural network, you have to put Flat between
    a pooling/convolution layer and an FC layer.

    Parameters
    ----------
    input_size : (int, int)
        The shape of input data expressed as (batch size, channels,
        rows, columns).

        The output shape will be (channels * rows * columns, batch size).
    name_allocator_func : zero-argument function, optional
        See Layer.__doc__.
    dim_allocator_func : one-argument function, optional
        See Layer.__doc__.
    generate_code : bool, optional
        See Layer.__doc__.
    """

    def __init__(self, input_size, name_allocator_func=alloc,
                 dim_allocator_func=dim_alloc, generate_code=False):
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
                               grid=gridR,
                               space_order=0, dtype=np.float64)

        return (None, I, R, None, None, output_grad, None)

    def execute(self, input_data):
        self._I.data[:] = input_data
        return super().execute()

    def equations(self):
        _, b, c, d = self._I.dimensions
        batch_size, channels, height, width = self._I.shape

        return ([Eq(self._R[b * height * width + c * height + d, a],
                    self._I[a, b, c, d]) for a in range(batch_size)], [])

    def backprop_equations(self, prev_layer, next_layer):
        layer = self

        prev_kernel_dims = prev_layer.kernel_gradients.dimensions
        dims = layer.result_gradients.dimensions

        batch_size, _, height, width = next_layer.result_gradients.shape
        next_dims = next_layer.result_gradients.dimensions

        return ([Inc(layer.result_gradients[dims[0], dims[1]],
                     prev_layer.kernel[prev_kernel_dims[0], dims[0]] *
                     prev_layer.result_gradients[prev_kernel_dims[0],
                                                 dims[1]])] +
                [Eq(next_layer.result_gradients[batch, next_dims[1],
                                                next_dims[2], next_dims[3]],
                    layer.result_gradients[next_dims[1] * height * width +
                                           next_dims[2] * height +
                                           next_dims[3], batch])
                 for batch in range(batch_size)] +
                next_layer.activation.backprop_eqs(next_layer), [])
