import joey
import numpy as np


def test_conv():
    pass


def test_max_pooling():
    pass


def test_flat():
    pass


def test_fully_connected():
    layer = joey.FullyConnected(weight_size=(3, 3), input_size=(3, 1))
    layer.kernel.data[:] = [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]
    output = layer.execute([[-1], [1], [-2]], [4, 1, -2])

    assert(np.array_equal(output, [[-1], [-10], [-19]]))
