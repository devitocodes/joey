import joey
import numpy as np
from joey.activation import ReLU


def test_conv():
    pass


def test_max_pooling():
    layer = joey.MaxPooling(kernel_size=(2, 2), input_size=(2, 2, 3, 3))
    output = layer.execute(np.array([[[[1, 2, 3],
                                       [4, 5, 6],
                                       [7, 8, 9]],
                                      [[10, 11, 12],
                                       [13, 14, 15],
                                       [16, 17, 18]]],
                                     [[[19, 20, 21],
                                       [22, 23, 24],
                                       [25, 26, 27]],
                                      [[28, 29, 30],
                                       [31, 32, 33],
                                       [34, 35, 36]]]]))

    assert(np.array_equal(output, [[[[5, 6],
                                     [8, 9]],
                                    [[14, 15],
                                     [17, 18]]],
                                   [[[23, 24],
                                     [26, 27]],
                                    [[32, 33],
                                     [35, 36]]]]))


def test_max_pooling_larger_stride():
    layer = joey.MaxPooling(kernel_size=(2, 2), input_size=(2, 2, 3, 3),
                            stride=(1, 2), strict_stride_check=False)
    output = layer.execute(np.array([[[[1, 2, 3],
                                       [4, 5, 6],
                                       [7, 8, 9]],
                                      [[10, 11, 12],
                                       [13, 14, 15],
                                       [16, 17, 18]]],
                                     [[[19, 20, 21],
                                       [22, 23, 24],
                                       [25, 26, 27]],
                                      [[28, 29, 30],
                                       [31, 32, 33],
                                       [34, 35, 36]]]]))

    assert(np.array_equal(output, [[[[5],
                                     [8]],
                                    [[14],
                                     [17]]],
                                   [[[23],
                                     [26]],
                                    [[32],
                                     [35]]]]))


def test_max_pooling_relu():
    layer = joey.MaxPooling(kernel_size=(2, 2), input_size=(2, 2, 3, 3),
                            activation=ReLU())
    output = layer.execute(np.array([[[[-1, -2, 3],
                                       [-4, -5, 6],
                                       [7, 8, 9]],
                                      [[10, 11, 12],
                                       [13, -14, -15],
                                       [16, -17, -18]]],
                                     [[[19, -20, -21],
                                       [22, -23, -24],
                                       [25, 26, 27]],
                                      [[28, 29, 30],
                                       [31, 32, 33],
                                       [34, 35, 36]]]]))

    assert(np.array_equal(output, [[[[0, 6],
                                     [8, 9]],
                                    [[13, 12],
                                     [16, 0]]],
                                   [[[22, 0],
                                     [26, 27]],
                                    [[32, 33],
                                     [35, 36]]]]))


def test_flat():
    layer = joey.Flat(input_size=(2, 2, 3, 3))
    output = layer.execute([[[[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]],
                             [[10, 11, 12],
                              [13, 14, 15],
                              [16, 17, 18]]],
                            [[[19, 20, 21],
                              [22, 23, 24],
                              [25, 26, 27]],
                             [[28, 29, 30],
                              [31, 32, 33],
                              [34, 35, 36]]]])

    assert(np.array_equal(output, [[1, 19],
                                   [2, 20],
                                   [3, 21],
                                   [4, 22],
                                   [5, 23],
                                   [6, 24],
                                   [7, 25],
                                   [8, 26],
                                   [9, 27],
                                   [10, 28],
                                   [11, 29],
                                   [12, 30],
                                   [13, 31],
                                   [14, 32],
                                   [15, 33],
                                   [16, 34],
                                   [17, 35],
                                   [18, 36]]))


def test_fully_connected():
    layer = joey.FullyConnected(weight_size=(3, 3), input_size=(3, 1))
    layer.kernel.data[:] = [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]
    output = layer.execute([[-1], [1], [-2]], [4, 1, -2])

    assert(np.array_equal(output, [[-1], [-10], [-19]]))


def test_fully_connected_relu():
    layer = joey.FullyConnected(weight_size=(3, 3), input_size=(3, 1),
                                activation=ReLU())
    layer.kernel.data[:] = [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]
    output = layer.execute([[-1], [1], [-2]], [6, 1, -2])

    assert(np.array_equal(output, [[1], [0], [0]]))
