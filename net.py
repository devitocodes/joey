import devito.ml as ml
import numpy as np
from devito import Eq, Inc, Operator, ConditionalDimension, Ne, Function, \
    Constant
from devito.ml import default_name_allocator as alloc
from devito.ml import default_dim_allocator as dim_alloc
from sympy import And


class Net:
    def __init__(self, layers: list):
        self._layers = layers
        self._batch_constant = Constant(name='batch', dtype=np.int32)
        self._backward_arg_dict = {}

        eqs = self._gen_eqs()
        backprop_eqs = self._gen_backprop_eqs()

        parameter_lists = list(map(ml.Layer.pytorch_parameters, self._layers))
        parameters = []

        for (kernel_parameter, bias_parameter) in parameter_lists:
            if kernel_parameter is not None:
                parameters.append(kernel_parameter)

            if bias_parameter is not None:
                parameters.append(bias_parameter)

        self._parameters = parameters
        self._init_parameters()

        self._forward_operator = Operator(eqs)
        self._backward_operator = Operator(backprop_eqs)

    def _init_parameters(self):
        for layer in self._layers:
            if layer.kernel is not None:
                layer.kernel.data[:] = np.random.rand(*layer.kernel.shape) - 0.5

            if layer.bias is not None:
                layer.bias.data[:] = np.random.rand(*layer.bias.shape) - 0.5

    def _gen_eqs(self):
        eqs = []
        input_function = None

        for layer in self._layers:
            eqs += layer.equations(input_function=input_function)
            input_function = layer.result

        return eqs

    def _gen_backprop_eqs(self):
        eqs = []

        for i in range(len(self._layers) - 1, -1, -1):
            if i < len(self._layers) - 1:
                prev_layer = self._layers[i + 1]
            else:
                prev_layer = None

            if i > 0:
                next_layer = self._layers[i - 1]
            else:
                next_layer = None

            layer = self._layers[i]

            if issubclass(type(layer), ml.FullyConnected):
                eqs += self._fully_connected_backprop_eqs(layer, prev_layer,
                                                          next_layer)
            elif type(layer) == ml.Subsampling:
                eqs += self._subsampling_backprop_eqs(layer, prev_layer,
                                                      next_layer)
            elif type(layer) == ml.Conv:
                eqs += self._conv_backprop_eqs(layer, prev_layer, next_layer)
            elif type(layer) == ml.Flat:
                eqs += self._flat_backprop_eqs(layer, prev_layer, next_layer)
            else:
                raise NotImplementedError

        return eqs

    def _fully_connected_backprop_eqs(self, layer, prev_layer, next_layer):
        dims = layer.result_gradients.dimensions
        bias_dims = layer.bias_gradients.dimensions
        kernel_dims = layer.kernel_gradients.dimensions

        if prev_layer is None:
            return [Inc(layer.bias_gradients[bias_dims[0]],
                    layer.result_gradients[bias_dims[0]]),
                    Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1]],
                        next_layer.result[kernel_dims[1],
                                          self._batch_constant] *
                        layer.result_gradients[kernel_dims[0]])]

        prev_dims = prev_layer.result_gradients.dimensions

        return [Eq(layer.result_gradients, 0),
                Inc(layer.result_gradients[dims[0]],
                    prev_layer.kernel[prev_dims[0], dims[0]] *
                    prev_layer.result_gradients[prev_dims[0]])] + \
            layer.activation.backprop_eqs(layer, self._batch_constant) + \
            [Inc(layer.bias_gradients[bias_dims[0]],
                 layer.result_gradients[bias_dims[0]]),
             Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1]],
                 next_layer.result[kernel_dims[1], self._batch_constant] *
                 layer.result_gradients[kernel_dims[0]])]

    def _subsampling_backprop_eqs(self, layer, prev_layer, next_layer):
        if next_layer is None:
            return []

        a, b = dim_alloc(2)
        self._backward_arg_dict[a.name + '_M'] = layer.kernel_size[0] - 1
        self._backward_arg_dict[b.name + '_M'] = layer.kernel_size[1] - 1
        processed = Function(name=alloc(), grid=layer.result.grid,
                             space_order=0, dtype=np.float64)

        dims = layer.result.dimensions

        # The first dimension corresponding to a batch index must be
        # discarded here.
        dims = dims[1:]

        stride_rows, stride_cols = layer.stride

        cd1 = ConditionalDimension(name=alloc(), parent=b,
                                   condition=And(Ne(processed[self._batch_constant,
                                                              dims[0],
                                                              dims[1],
                                                              dims[2]], 1),
                                                 ~Ne(next_layer
                                                     .result[self._batch_constant,
                                                             dims[0],
                                                             stride_rows *
                                                             dims[1] + a,
                                                             stride_cols *
                                                             dims[2] + b],
                                                     layer.result[self._batch_constant,
                                                                  dims[0],
                                                                  dims[1],
                                                                  dims[2]])))

        return [Eq(next_layer.result_gradients[dims[0], stride_rows * dims[1] +
                                               a, stride_cols * dims[2] + b],
                   layer.result_gradients[dims[0], dims[1], dims[2]],
                   implicit_dims=cd1),
                Eq(processed[self._batch_constant, dims[0], dims[1], dims[2]],
                   1, implicit_dims=(a, b, cd1))] + \
            next_layer.activation.backprop_eqs(next_layer,
                                               self._batch_constant)

    def _conv_backprop_eqs(self, layer, prev_layer, next_layer):
        kernel_dims = layer.kernel_gradients.dimensions
        bias_dims = layer.bias_gradients.dimensions
        dims = layer.result_gradients.dimensions

        eqs = [Inc(layer.bias_gradients[bias_dims[0]],
                   layer.result_gradients[bias_dims[0], dims[1], dims[2]])]

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

            eqs += [Inc(layer.kernel_gradients[kernel_dims[0], kernel_dims[1],
                                               kernel_dims[2], kernel_dims[3]],
                        layer.result_gradients[kernel_dims[0], dims[1],
                                               dims[2]] *
                        next_layer.result[self._batch_constant, kernel_dims[1],
                                          kernel_dims[2] + dims[1],
                                          kernel_dims[3] + dims[2]]),
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
                                                   self._batch_constant)
        else:
            eqs.append(Inc(layer.kernel_gradients[kernel_dims[0],
                                                  kernel_dims[1],
                                                  kernel_dims[2],
                                                  kernel_dims[3]],
                           layer.result_gradients[kernel_dims[0], dims[1],
                                                  dims[2]] *
                           layer.input[self._batch_constant, kernel_dims[1],
                                       kernel_dims[2] + dims[1],
                                       kernel_dims[3] + dims[2]]))

        return eqs

    def _flat_backprop_eqs(self, layer, prev_layer, next_layer):
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
            next_layer.activation.backprop_eqs(next_layer,
                                               self._batch_constant)

    @property
    def pytorch_parameters(self):
        return self._parameters

    def forward(self, input_data):
        self._layers[0].input.data[:] = input_data
        self._forward_operator.apply()
        return self._layers[-1].result.data

    def backward(self, loss_gradient_func, pytorch_optimizer=None):
        if len(self._layers[-1].result.shape) < 2:
            batch_size = 1
        else:
            batch_size = self._layers[-1].result.shape[1]

        for i in range(batch_size):
            self._batch_constant.data = i
            self._layers[-1].result_gradients.data[:] = \
                loss_gradient_func(self._layers[-1], i)
            self._backward_operator.apply(**self._backward_arg_dict)

        if pytorch_optimizer is not None:
            pytorch_optimizer.step()
