from abc import ABC, abstractmethod
from devito import Operator, Function
from numpy import array


class Layer(ABC):
    @abstractmethod
    def _allocate(self) -> Function:
        # This method should return a Function object corresponding to
        # an output of the layer.
        pass

    def setup(self, input_data):
        self._input_data = input_data
        self._R = self._allocate()
        self._op = Operator(self.equations())
        self._op.cfunction

    def execute(self) -> array:
        self._op.apply()
        return self._R.data

    @abstractmethod
    def equations(self) -> list:
        pass
