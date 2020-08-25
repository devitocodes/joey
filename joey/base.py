from abc import ABC, abstractmethod
from devito import Operator, Function, dimensions
from joey import Activation
from joey import activation as activ
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
                 input_size, activation=activ.Dummy(),
                 name_allocator_func=default_name_allocator,
                 dim_allocator_func=default_dim_allocator,
                 generate_code=True):
        if activation is None:
            activation = activ.Dummy()

        if not issubclass(type(activation), Activation):
            raise Exception("activation must be an instance of Activation or "
                            "its subclass")

        self._activation = activation

        self._K, self._I, self._R, self._bias, self._KG, self._RG, \
            self._biasG = self._allocate(kernel_size,
                                         input_size,
                                         name_allocator_func,
                                         dim_allocator_func)

        if generate_code:
            eqs, args = self.equations()
            self._arg_dict = dict(args)
            self._op = Operator(eqs)
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
    def result_gradients(self):
        return self._RG

    @property
    def bias_gradients(self):
        return self._biasG

    @property
    def activation(self):
        return self._activation

    def pytorch_parameters(self):
        from torch import from_numpy
        from torch.nn import Parameter

        kernel_parameter = None
        bias_parameter = None

        if self._K is not None:
            kernel_tensor = from_numpy(self._K.data)
            kernel_parameter = Parameter(kernel_tensor, requires_grad=False)

            if self._KG is not None:
                kernel_parameter.grad = from_numpy(self._KG.data)

        if self._bias is not None:
            bias_tensor = from_numpy(self._bias.data)
            bias_parameter = Parameter(bias_tensor, requires_grad=False)

            if self._biasG is not None:
                bias_parameter.grad = from_numpy(self._biasG.data)

        return (kernel_parameter, bias_parameter)

    @abstractmethod
    def _allocate(self, kernel_size, input_size, name_allocator_func,
                  dim_allocator_func) -> (Function, Function, Function,
                                          Function, Function, Function,
                                          Function):
        # This method should return a (Function, Function, Function, Function,
        # Function, Function, Function) object corresponding to a kernel,
        # input, output, bias, kernel gradients, output gradients and bias
        # gradients of the layer respectively.
        #
        # Kernel, output and bias gradients are for backpropagation purposes.
        pass

    @abstractmethod
    def execute(self, kernel_data=None, input_data=None, bias=None) -> array:
        self._op.apply(**self._arg_dict)
        return self._R.data

    @abstractmethod
    def equations(self) -> (list, list):
        pass

    @abstractmethod
    def backprop_equations(self, prev_layer, next_layer,
                           batch_constant) -> (list, list):
        pass
