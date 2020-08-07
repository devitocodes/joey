from abc import ABC, abstractmethod


class Activation(ABC):
    def __init__(self, function):
        self._function = function

    def __call__(self, *args, **kwargs):
        return self._function(*args, **kwargs)

    @abstractmethod
    def backprop_eqs(self, layer):
        pass
