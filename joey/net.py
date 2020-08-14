import joey as ml
import numpy as np
from devito import Eq, Inc, Operator, ConditionalDimension, Ne, Function, \
    Constant
from joey import default_name_allocator as alloc
from joey import default_dim_allocator as dim_alloc
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

            eqs += self._layers[i].backprop_equations(prev_layer, next_layer,
                                                      self._batch_constant,
                                                      self._backward_arg_dict)

        return eqs

    @property
    def pytorch_parameters(self):
        return self._parameters

    def forward(self, input_data):
        for layer in self._layers:
            layer.result.data[:] = 0

        self._layers[0].input.data[:] = input_data
        self._forward_operator.apply()
        return self._layers[-1].result.data

    def backward(self, loss_gradient_func, pytorch_optimizer=None):
        for layer in self._layers:
            if layer.kernel_gradients is not None:
                layer.kernel_gradients.data[:] = 0

            if layer.bias_gradients is not None:
                layer.bias_gradients.data[:] = 0

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
