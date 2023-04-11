# Joey
Joey is a machine learning framework running on top of [Devito](https://github.com/devitocodes/devito) and using [PyTorch](https://github.com/pytorch/pytorch) optimizers. It is currently under development as a research project.

## Supported features
* Creating and running a standalone neural network layer
* Creating a neural network consisting of several layers
* A forward pass through a neural network with batch processing
* A backward pass through a neural network with batch processing
* Producing backpropagation equations automatically based on the list of layers in a neural network (only a loss function must be defined manually by the user)
* Training a neural network with PyTorch optimizers
* Initiate a transformer neural network for image classification

Unlike other machine learning frameworks, Joey generates and compiles an optimized low-level code on-the-spot (using Devito) for both standalone layers and proper neural networks.

## Supported neural network layers
* 2D convolution
* 2D max pooling (other types of 2D pooling can be implemented by the user by extending the `Pooling` abstract class)
* Full connection
* Flattening (an internal layer turning 2D data with channels into a 1D vector or 2D matrix, depending on the batch size)
* 3D FullyConnected
* Einsun function
* Dropout 1, 2, 3 and 4 dimensions
* Norm 2D
* Norm 3D
* Softmax 3D and 4D function

## Supported modules
* MultiHeadAttention
* VisionEnconder

## Built-in Models
* ViT (Vision Transformer)

## Supported activation functions
* ReLU
* Softmax (only via the `FullyConnectedSoftmax` class or with `Softmax3d` / `Softmax4d` function)
* Dummy (`f(x) = x`)

Other activation functions can be implemented by extending the `Activation` abstract class.

## Installation
### Docker
1. Clone the repository: `git clone https://github.com/devitocodes/joey`
2. Change your working directory to where you have cloned the repository.
3. Build a Docker image (only a CPU version is provided at the moment): `docker build -f Dockerfile_CPU -t devitocodes/joey:latest .`
4. Done! If you want to start a Python interpreter with Joey, run `docker run -i -t devitocodes/joey:latest python`.

### Manually (recommended for contributors)
*Please note that this method installs PyTorch with CUDA by default. If you want only a CPU version of PyTorch, you should install it by yourself beforehand.*

1. Make sure you are in the environment which you want to install Joey in (e.g. a Python/conda virtual environment).
2. Clone the repository: `git clone https://github.com/devitocodes/joey`
3. Install Joey in editable mode: `pip install -e <path to where you have cloned the repository>`

Done! You can now use Joey in your environment. If you want to make changes to the Joey code, you can do so in the directory where you have cloned the repository.

### PyPI
Joey is not available on PyPI yet.

## How to use
To start working with Joey, import the following packages:
```
import joey
import joey.activation  # If you want to use activation in neural network layers
```

Afterwards, you are free to use all functions Joey offers. The recommended way of getting started is going through examples inside the `examples` directory in this repository and looking at `__doc__` that is provided in every Joey class and public/abstract class method.
