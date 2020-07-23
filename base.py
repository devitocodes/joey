from abc import ABC, abstractmethod
from devito import Operator, Function, dimensions
from numpy import array

index = 0
dim_index = 0


def default_name_allocator():
    global index
    name = 'f' + str(index)
    index += 1
    return name


def default_dim_allocator(count):
    global dim_index
    names = ''
    for i in range(count):
        names += 'd' + str(dim_index) + ' '
        dim_index += 1
    names = names[:-1]
    return dimensions(names)


class Layer(ABC):
    def __init__(self, kernel_size,
                 input_size, name_allocator_func=default_name_allocator,
                 dim_allocator_func=default_dim_allocator,
                 generate_code=True):
        self._K, self._I, self._R, self._bias, self._KG, self._IG = \
            self._allocate(kernel_size,
                           input_size,
                           name_allocator_func,
                           dim_allocator_func)

        if generate_code:
            self._op = Operator(self.equations())
            self._op.cfunction

    @property
    def kernel(self):
        return self._K

    @property
    def input(self):
        return self._I

    @property
    def result(self):
        return self._R

    @property
    def bias(self):
        return self._bias

    @property
    def kernel_gradients(self):
        return self._KG

    @property
    def input_gradients(self):
        return self._IG

    @abstractmethod
    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func) -> (Function, Function, Function,
                                          Function, Function, Function):
        # This method should return a
        # (Function, Function, Function, Function, Function, Function)
        # object corresponding to a kernel, input, output, bias, kernel
        # gradients and input gradients of the layer respectively.
        #
        # Kernel and input gradients are for backpropagation purposes.
        pass

    @abstractmethod
    def execute(self, kernel_data=None, input_data=None, bias=None) -> array:
        self._op.apply()
        return self._R.data

    @abstractmethod
    def equations(self, input_function=None) -> list:
        pass


class LossFunction:
    def __init__(self, calculate_func):
        self._sum = 0
        self._count = 0
        self._calculate = calculate_func

    def add(self, expected, actual):
        result = self._calculate(expected, actual)
        self._sum += result
        self._count += 1

    def average(self):
        return self._sum / self._count
