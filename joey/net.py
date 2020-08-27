import joey as ml
import numpy as np
from devito import Eq, Operator


class Net:
    def __init__(self, layers: list):
        self._layers = layers
        self._forward_arg_dict = {}
        self._backward_arg_dict = {}

        eqs, args = self._gen_eqs()
        backprop_eqs, backprop_args = self._gen_backprop_eqs()

        for (key, value) in args:
            self._forward_arg_dict[key] = value

        for (key, value) in backprop_args:
            self._backward_arg_dict[key] = value

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

        self._forward_operator.cfunction
        self._backward_operator.cfunction

    def _init_parameters(self):
        for layer in self._layers:
            if layer.kernel is not None:
                layer.kernel.data[:] = \
                    np.random.rand(*layer.kernel.shape) - 0.5

            if layer.bias is not None:
                layer.bias.data[:] = np.random.rand(*layer.bias.shape) - 0.5

    def _gen_eqs(self):
        eqs = []
        args = []

        input_function = None

        for layer in self._layers:
            if input_function is not None:
                dims = input_function.dimensions
                eqs.append(Eq(layer.input[dims], input_function[dims]))

            layer_eqs, layer_args = layer.equations()

            args += layer_args
            eqs += layer_eqs
            input_function = layer.result

        return (eqs, args)

    def _gen_backprop_eqs(self):
        eqs = []
        args = []

        for i in range(len(self._layers) - 1, -1, -1):
            if i < len(self._layers) - 1:
                prev_layer = self._layers[i + 1]
            else:
                prev_layer = None

            if i > 0:
                next_layer = self._layers[i - 1]
            else:
                next_layer = None

            layer_eqs, layer_args = \
                self._layers[i].backprop_equations(prev_layer, next_layer)

            args += layer_args
            eqs += layer_eqs

        return (eqs, args)

    @property
    def pytorch_parameters(self):
        return self._parameters

    def forward(self, input_data):
        for layer in self._layers:
            layer.result.data[:] = 0

        self._layers[0].input.data[:] = input_data
        self._forward_operator.apply(**self._forward_arg_dict)
        return self._layers[-1].result.data

    def backward(self, loss_gradient_func, pytorch_optimizer=None):
        for layer in self._layers:
            if layer.kernel_gradients is not None:
                layer.kernel_gradients.data[:] = 0

            if layer.bias_gradients is not None:
                layer.bias_gradients.data[:] = 0

            if layer.result_gradients is not None:
                layer.result_gradients.data[:] = 0

        batch_size = self._layers[-1].result.shape[1]

        self._layers[-1].result_gradients.data[:] = \
            np.transpose(np.array(loss_gradient_func(self._layers[-1])))
        self._backward_operator.apply(**self._backward_arg_dict)

        for layer in self._layers:
            if layer.kernel_gradients is not None:
                layer.kernel_gradients.data[:] /= batch_size

            if layer.bias_gradients is not None:
                layer.bias_gradients.data[:] /= batch_size

        if pytorch_optimizer is not None:
            pytorch_optimizer.step()
