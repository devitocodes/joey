from abc import ABC, abstractmethod
from sympy import Max, sign
from devito import Eq


class Activation(ABC):
    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @abstractmethod
    def backprop_eqs(self, layer):
        pass


class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: Max(0, x))

    def backprop_eqs(self, layer):
        return [Eq(layer.result_gradients,
                   layer.result_gradients * sign(layer.result))]


class Dummy(Activation):
    def __init__(self):
        super().__init__(lambda x: x)

    def backprop_eqs(self, layer):
        return []
