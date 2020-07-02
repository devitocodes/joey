from abc import ABC, abstractmethod
from devito import Operator, Function
from numpy import array


class Layer(ABC):
    def __init__(self, input_data):
        self._input_data = input_data
        self._R = self._allocate()

    @abstractmethod
    def _allocate(self) -> Function:
        # This method should return a Function object corresponding to
        # an output of the layer.
        pass

    @abstractmethod
    def execute(self) -> (Operator, array):
        pass

    @abstractmethod
    def equations(self) -> list:
        pass
