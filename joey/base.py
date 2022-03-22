from abc import ABC, abstractmethod
from devito import Operator, Function, SpaceDimension
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
    return [SpaceDimension(x) for x in names]


class Layer(ABC):
    """
    An abstract class representing a neural network layer.

    When you create a subclass of Layer, you must implement the following
    methods: _allocate(), equations(), backprop_equations() and execute().

    Parameters
    ----------
    kernel_size : tuple of ints
        The shape of a kernel/weight Function (represented internally by
        a NumPy array).
    input_size : tuple of ints
        The shape of an input data Function.
    activation : Activation, optional
        An activation function to be applied after processing the input data.
        The default value is a Dummy object corresponding to f(x) = x.
    name_allocator_func : zero-argument function, optional
        A function allocating unique names for Function objects. The default
        value is default_name_allocator() producing 'f0', 'f1', 'f2' etc.
    dim_allocator_func : one-argument function, optional
        A function allocating unique dimension names. It must accept one
        argument describing how many dimensions should be returned. The
        default value is default_dim_allocator() producing 'd0', 'd1', 'd2'
        etc.
    generate_code : bool, optional
        A boolean indicating whether a low-level code should be produced
        strictly for this layer. If set to True, it will be possible to
        execute the layer alone using execute(). The default value is
        False and should not be changed if you want to use the layer only as
        part of a network.
    """

    def __init__(self, kernel_size,
                 input_size, activation=activ.Dummy(),
                 name_allocator_func=default_name_allocator,
                 dim_allocator_func=default_dim_allocator,
                 generate_code=False):
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
        """A Function object corresponding to a kernel/weight array."""
        return self._K

    @property
    def input(self):
        """A Function object corresponding to an input data array."""
        return self._I

    @property
    def result(self):
        """A Function object corresponding to a result array."""
        return self._R

    @property
    def bias(self):
        """A Function object corresponding to a bias array."""
        return self._bias

    @property
    def kernel_gradients(self):
        """A Function object corresponding to a kernel gradients array."""
        return self._KG

    @property
    def result_gradients(self):
        """A Function object corresponding to an output gradients array."""
        return self._RG

    @property
    def bias_gradients(self):
        """A Function object corresponding to a bias gradients array."""
        return self._biasG

    @property
    def activation(self):
        """An Activation object corresponding to the activation function."""
        return self._activation

    def pytorch_parameters(self):
        """
        Returns a tuple (kernel, bias) of parameters
        required by a PyTorch optimizer.
        """
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
        """
        This method should return a (Function, Function, Function, Function,
        Function, Function, Function) object corresponding to a kernel,
        input, output, bias, kernel gradients, output gradients and bias
        gradients of the layer respectively.

        Kernel, output and bias gradients are for backpropagation purposes.
        """
        pass

    @abstractmethod
    def execute(self, kernel_data=None, input_data=None, bias=None) -> array:
        """
        Runs a forward pass through the layer and returns its output. This
        method will work provided that the layer has been instantiated with
        generate_code=True. Otherwise, an error related to accessing
        a None object or a non-existing attribute will be thrown.

        Parameters
        ----------
        kernel_data : np.ndarray, optional
            A NumPy array representing kernel/weight parameters. The default
            value is None.
        input_data : np.ndarray, optional
            A NumPy array representing input data. The default value is None.
        bias : np.ndarray, optional
            A NumPy array representing bias. The default value is None.
        """
        self._op.apply(**self._arg_dict)
        return self._R.data

    @abstractmethod
    def equations(self) -> (list, list):
        """
        Returns a two-tuple of lists. The first list consists of Devito
        equations describing how a forward pass through the layer should work.
        The second list consists of (key, value) pairs describing what keyword
        arguments should be passed to Operator.apply() when running the forward
        pass.

        When implementing this method, the following fields may be useful:
        * self._I: corresponds to self.input
        * self._K: corresponds to self.kernel
        * self._R: corresponds to self.result
        * self._bias: corresponds to self.bias
        """
        pass

    @abstractmethod
    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        """
        Returns a two-tuple of lists. The first list consists of Devito
        equations describing how a backward pass through the layer should work.
        The second list consists of (key, value) pairs describing what keyword
        arguments should be passed to Operator.apply() when running
        the backward pass.

        When implementing this method, the following fields may be useful:
        * self._KG: corresponds to self.kernel_gradients
        * self._RG: corresponds to self.result_gradients
        * self._biasG: corresponds to self.bias_gradients

        Parameters
        ----------
        prev_layer : Layer
            The previous layer in a backpropagation chain.
        next_layer : Layer
            The next layer in a backpropagation chain.

        Please note that the current layer in a backpropagation chain is
        represented by self.
        """
        pass
