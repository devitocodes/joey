import joey
import numpy as np
from joey.activation import ReLU


def test_conv():
    pass


def test_max_pooling():
    pass


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
