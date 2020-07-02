from abc import ABC, abstractmethod
from devito import Operator
from numpy import array


class Layer(ABC):
    def __init__(self, input_data):
        self._input_data = input_data

    @abstractmethod
    def execute(self) -> (Operator, array):
        pass

    @abstractmethod
    def equations(self) -> list:
        pass
