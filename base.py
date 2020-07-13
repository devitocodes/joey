from abc import ABC, abstractmethod
from devito import Operator, Function
from numpy import array

index = 0


def default_name_allocator():
    global index
    name = 'f' + str(index)
    index += 1
    return name


class Layer(ABC):
    def __init__(self, kernel_size,
                 input_size, name_allocator_func=default_name_allocator,
                 generate_code=True):
        self._K, self._I, self._R = self._allocate(kernel_size,
                                                   input_size,
                                                   name_allocator_func)

        if generate_code:
            self._op = Operator(self.equations())
            self._op.cfunction

    @abstractmethod
    def _allocate(self, kernel_size, input_size,
                  name_allocator_func) -> (Function, Function, Function):
        # This method should return a (Function, Function, Function) triple
        # corresponding to a kernel, input and output of the layer
        # respectively.
        pass

    @abstractmethod
    def execute(self, kernel_data=None, input_data=None, bias=None) -> array:
        self._op.apply()
        return self._R.data

    @abstractmethod
    def equations(self, input_function=None) -> list:
        pass
