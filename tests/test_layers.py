import joey
import numpy as np
from joey.activation import ReLU
from devito import logger
from utils import get_run_count

logger.set_log_noperf()


def test_conv():
    layer = joey.Conv(kernel_size=(2, 2, 2), input_size=(2, 2, 3, 3),
                      generate_code=True)
    layer.kernel.data[:] = [[[[1, -2],
                              [3, 4]],
                             [[5, 6],
                              [7, 8.5]]],
                            [[[-9.5, 10],
                              [11, 12]],
                             [[13, -14],
                              [-15, 16]]]]

    for i in range(get_run_count()):
        output = layer.execute(np.array([[[[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]],
                                          [[0, 0, 1],
                                           [0, 1, 0],
                                           [0, 0, 2]]],
                                         [[[-1, -2, -3],
                                           [4, 6, 8],
                                           [11, 0, 2]],
                                          [[9, 8, 7],
                                           [6, 5, 4],
                                           [3, 2, 1]]]]), [0, 0])

        assert(np.array_equal(output, [[[[37.5, 48],
                                         [53, 75]],
                                        [[130.5, 109],
                                         [171, 253.5]]],
                                       [[[216.5, 205],
                                         [123, 69.5]],
                                        [[100.5, 146],
                                         [138, 42]]]]))


def test_conv_relu():
    layer = joey.Conv(kernel_size=(2, 2, 2), input_size=(2, 2, 3, 3),
                      activation=ReLU(), generate_code=True)
    layer.kernel.data[:] = [[[[1, -2],
                              [3, 4]],
                             [[5, 6],
                              [7, 8.5]]],
                            [[[-9.5, 10],
                              [11, 12]],
                             [[13, -14],
                              [-15, 16]]]]

    for i in range(get_run_count()):
        output = layer.execute(np.array([[[[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]],
                                          [[0, 0, 1],
                                           [0, 1, 0],
                                           [0, 0, 2]]],
                                         [[[-1, -2, -3],
                                           [4, 6, 8],
                                           [11, 0, 2]],
                                          [[9, 8, 7],
                                           [6, 5, 4],
                                           [3, 2, 1]]]]), [-50, -79.75])

        assert(np.array_equal(output, [[[[0, 0],
                                         [3, 25]],
                                        [[50.75, 29.25],
                                         [91.25, 173.75]]],
                                       [[[166.5, 155],
                                         [73, 19.5]],
                                        [[20.75, 66.25],
                                         [58.25, 0]]]]))


def test_conv_larger_stride():
    layer = joey.Conv(kernel_size=(2, 2, 2), input_size=(2, 2, 3, 3),
                      stride=(2, 1), strict_stride_check=False,
                      generate_code=True)
    layer.kernel.data[:] = [[[[1, -2],
                              [3, 4]],
                             [[5, 6],
                              [7, 8.5]]],
                            [[[-9.5, 10],
                              [11, 12]],
                             [[13, -14],
                              [-15, 16]]]]

    for i in range(get_run_count()):
        output = layer.execute(np.array([[[[1, 2, 3],
                                           [4, 5, 6],
                                           [7, 8, 9]],
                                          [[0, 0, 1],
                                           [0, 1, 0],
                                           [0, 0, 2]]],
                                         [[[-1, -2, -3],
                                           [4, 6, 8],
                                           [11, 0, 2]],
                                          [[9, 8, 7],
                                           [6, 5, 4],
                                           [3, 2, 1]]]]), [0, 0])

        assert(np.array_equal(output, [[[[37.5, 48]],
                                        [[130.5, 109]]],
                                       [[[216.5, 205]],
                                        [[100.5, 146]]]]))


def test_max_pooling():
    layer = joey.MaxPooling(kernel_size=(2, 2), input_size=(2, 2, 3, 3),
                            generate_code=True)

    for i in range(get_run_count()):
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
                            stride=(1, 2), strict_stride_check=False,
                            generate_code=True)

    for i in range(get_run_count()):
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
                            activation=ReLU(), generate_code=True)

    for i in range(get_run_count()):
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
    layer = joey.Flat(input_size=(2, 2, 3, 3), generate_code=True)

    for i in range(get_run_count()):
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
    layer = joey.FullyConnected(weight_size=(3, 3), input_size=(3, 1),
                                generate_code=True)
    layer.kernel.data[:] = [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]

    for i in range(get_run_count()):
        output = layer.execute([[-1], [1], [-2]], [4, 1, -2])

        assert(np.array_equal(output, [[-1], [-10], [-19]]))


def test_fully_connected_relu():
    layer = joey.FullyConnected(weight_size=(3, 3), input_size=(3, 1),
                                activation=ReLU(), generate_code=True)
    layer.kernel.data[:] = [[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]]

    for i in range(get_run_count()):
        output = layer.execute([[-1], [1], [-2]], [6, 1, -2])

        assert(np.array_equal(output, [[1], [0], [0]]))
