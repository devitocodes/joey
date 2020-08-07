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

        return [Eq(layer.result_gradients[dims[0], dims[1], dims[2]],
                   layer.result_gradients[dims[0], dims[1], dims[2]] *
                   sign(layer.result[batch_index, dims[0], dims[1], dims[2]]))]


class Dummy(Activation):
    def __init__(self):
        super().__init__(lambda x: None)

    def backprop_eqs(self, layer, batch_index):
        return []
