from devito.ml import Layer
from devito import Grid, Function, dimensions, Eq, Inc
import numpy as np


class Conv(Layer):
    def __init__(self, kernel, input_data):
        self._error_check(kernel, input_data)

        self._kernel = kernel
        super().__init__(input_data=input_data)

    def _error_check(self, kernel, input_data):
        if kernel is None or len(kernel) == 0:
            raise Exception("kernel must not be empty")

        if input_data is None or len(input_data) == 0:
            raise Exception("image must not be empty")

        different_row_length = False
        for row in kernel:
            if len(row) != len(kernel[0]):
                different_row_length = True
                break

        if different_row_length:
            raise Exception("kernel has an invalid shape")

        different_row_length = False
        for row in input_data:
            if len(row) != len(input_data[0]):
                different_row_length = True
                break

        if different_row_length:
            raise Exception("image has an invalid shape")

        if len(kernel) % 2 == 0 or len(kernel[0]) % 2 == 0:
            raise Exception("The dimensions of kernel must be odd")

    def _allocate(self):
        gridA = Grid(shape=(len(self._kernel), len(self._kernel[0])),
                     dimensions=dimensions('m n'))
        A = Function(name='A', grid=gridA, space_order=0)
        A.data[:] = self._kernel

        gridBR = Grid(shape=(len(self._input_data), len(self._input_data[0])))

        B = Function(name='B', grid=gridBR, space_order=1)
        B.data[:] = self._input_data

        self._A = A
        self._B = B

        R = Function(name='R', grid=gridBR, space_order=0)
        return R

    def equations(self):
        x, y = self._B.dimensions
        kernel_rows, kernel_cols = self._A.shape

        return [Eq(self._R[x, y],
                   sum([self._A[kernel_rows - i - 1,
                                kernel_cols - j - 1] *
                        self._B[x - kernel_rows // 2 + i,
                                y - kernel_cols // 2 + j]
                        for i in range(kernel_rows)
                        for j in range(kernel_cols)]))]


class Subsampling(Layer):
    def __init__(self, kernel_size, feature_map, function,
                 stride=(1, 1), padding=(0, 0)):
        # All sizes are expressed as (rows, columns).

        self._error_check(kernel_size, feature_map, stride, padding)

        self._kernel_size = kernel_size
        self._function = function
        self._stride = stride
        self._padding = padding

        super().__init__(input_data=feature_map)

    def _error_check(self, kernel_size, feature_map, stride, padding):
        if feature_map is None or len(feature_map) == 0:
            raise Exception("Feature map must not be empty")

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

        map_height = len(feature_map) + 2 * padding[0]
        map_width = len(feature_map[0]) + 2 * padding[1]
        kernel_height, kernel_width = kernel_size

        if (map_height - kernel_height) % stride[0] != 0 or \
           (map_width - kernel_width) % stride[1] != 0:
            raise Exception("Stride " + str(stride) + " is not "
                            "compatible with feature map, kernel and padding "
                            "sizes")

    def _allocate(self):
        map_height = len(self._input_data) + 2 * self._padding[0]
        map_width = len(self._input_data[0]) + 2 * self._padding[1]
        kernel_height, kernel_width = self._kernel_size

        gridB = Grid(shape=(map_height, map_width))
        B = Function(name='B', grid=gridB, space_order=0)

        a, b = dimensions('a b')
        gridR = Grid(shape=((map_height - kernel_height + self._stride[0])
                            // self._stride[0],
                            (map_width - kernel_width + self._stride[1])
                            // self._stride[1]),
                     dimensions=(a, b))
        R = Function(name='R', grid=gridR, space_order=0)

        for i in range(self._padding[0], map_height - self._padding[0]):
            B.data[i] = \
                np.concatenate(([0] * self._padding[1],
                                self._input_data[i - self._padding[0]],
                                [0] * self._padding[1]))

        self._B = B

        return R

    def equations(self):
        a, b = self._B.dimensions
        kernel_height, kernel_width = self._kernel_size

        return [Eq(self._R[a, b],
                   self._function([self._B[self._stride[0] * a + i,
                                           self._stride[1] * b + j]
                                   for i in range(kernel_height)
                                   for j in range(kernel_width)]))]


class FullyConnected(Layer):
    def __init__(self, weights, input_data):
        self._weights = weights
        super().__init__(input_data=input_data)

    def _allocate(self):
        weight_size = (len(self._weights), len(self._weights[0]))

        try:
            input_size = (len(self._input_data), len(self._input_data[0]))
            is_vector = False
        except TypeError:
            input_size = len(self._input_data)
            is_vector = True

        a, b, c = dimensions('a b c')

        gridW = Grid(shape=weight_size, dimensions=(a, b))
        W = Function(name='W', grid=gridW, space_order=0)

        if is_vector:
            gridV_dimensions = (b,)
            gridR_dimensions = (a,)
            gridR_shape = weight_size[0]
        else:
            gridV_dimensions = (b, c)
            gridR_dimensions = (a, c)
            gridR_shape = (weight_size[0], input_size[1])

        gridV = Grid(shape=input_size, dimensions=gridV_dimensions)
        V = Function(name='V', grid=gridV, space_order=0)

        W.data[:] = self._weights
        V.data[:] = self._input_data

        gridR = Grid(shape=gridR_shape, dimensions=gridR_dimensions)
        R = Function(name='R', grid=gridR, space_order=0)

        self._W = W
        self._V = V

        return R

    def equations(self):
        return [Inc(self._R, self._W * self._V)]
