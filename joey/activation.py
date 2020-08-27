from abc import ABC, abstractmethod
from sympy import Max, sign
from devito import Eq


class Activation(ABC):
    """
    An abstract class representing an activation function.

    When you create a subclass of Activation, you must implement
    the backprop_eqs() method.

    Parameters
    ----------
    function : function
        A function to apply to data. Usually, it will be a one-argument
        function f(x) where x is the raw output of a layer.
    """

    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @abstractmethod
    def backprop_eqs(self, layer):
        """
        Returns a list of Devito equations describing how backpropagation
        should proceed when encountering this activation function.

        Parameters
        ----------
        layer : Layer
            The next layer in a backpropagation chain.
        """
        pass


class ReLU(Activation):
    """An Activation subclass corresponding to ReLU."""
    def __init__(self):
        super().__init__(lambda x: Max(0, x))

    def backprop_eqs(self, layer):
        return [Eq(layer.result_gradients,
                   layer.result_gradients * sign(layer.result))]


class Dummy(Activation):
    """An Activation subclass corresponding to f(x) = x."""
    def __init__(self):
        super().__init__(lambda x: x)

    def backprop_eqs(self, layer):
        return []
