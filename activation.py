from abc import ABC, abstractmethod
from sympy import Max, sign
from devito import Eq


class Activation(ABC):
    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @abstractmethod
    def backprop_eqs(self, layer, batch_index):
        pass


class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: Max(0, x))

    def backprop_eqs(self, layer, batch_index):
        dims = layer.result_gradients.dimensions

        if len(dims) == 3:
            return [Eq(layer.result_gradients[dims[0], dims[1], dims[2]],
                       layer.result_gradients[dims[0], dims[1], dims[2]] *
                       sign(layer.result[batch_index, dims[0], dims[1],
                                         dims[2]]))]
        elif len(dims) == 2:
            return [Eq(layer.result_gradients[dims[0], dims[1]],
                       layer.result_gradients[dims[0], dims[1]] *
                       sign(layer.result[batch_index, dims[0], dims[1]]))]
        elif len(dims) == 1:
            return [Eq(layer.result_gradients[dims[0]],
                       layer.result_gradients[dims[0]] *
                       sign(layer.result[dims[0], batch_index]))]
        else:
            raise NotImplementedError


class Dummy(Activation):
    def __init__(self):
        super().__init__(lambda x: x)

    def backprop_eqs(self, layer, batch_index):
        return []
