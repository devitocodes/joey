import pytest
import joey
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from joey.activation import ReLU
from utils import compare


# PyTorch class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(2, 2, 2)
        self.fc = nn.Linear(8, 3)

    def forward(self, x):
        x = F.max_pool2d(x, 2, stride=(1, 1))
        x = F.relu(self.conv(x))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Helper functions
SEED = 282757891


@pytest.fixture
def net_arguments():
    np.random.seed(SEED)

    layer1 = joey.MaxPooling(kernel_size=(2, 2),
                             input_size=(1, 2, 4, 4),
                             generate_code=False)
    layer2 = joey.Conv(kernel_size=(2, 2, 2),
                       input_size=(1, 2, 3, 3),
                       activation=ReLU(),
                       generate_code=False)
    layer_flat = joey.Flat(input_size=(1, 2, 2, 2),
                           generate_code=False)
    layer3 = joey.FullyConnectedSoftmax(weight_size=(3, 8),
                                        input_size=(8, 1),
                                        generate_code=False)

    layers = [layer1, layer2, layer_flat, layer3]

    net = joey.Net(layers)

    pytorch_net = Net()
    pytorch_net.double()

    with torch.no_grad():
        pytorch_net.conv.weight[:] = torch.from_numpy(layer2.kernel.data)
        pytorch_net.conv.bias[:] = torch.from_numpy(layer2.bias.data)

        pytorch_net.fc.weight[:] = torch.from_numpy(layer3.kernel.data)
        pytorch_net.fc.bias[:] = torch.from_numpy(layer3.bias.data)

    return (net, pytorch_net, layers)


# Proper test functions
def test_forward_pass(net_arguments):
    net, pytorch_net, layers = net_arguments
    input_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]],
                            [[-1, -2, 0, 1],
                             [-2, -3, 1, 2],
                             [3, 4, 2, -1],
                             [-2, -3, -4, 9]]]],
                          dtype=np.float64)

    outputs = net.forward(input_data)
    pytorch_outputs = pytorch_net(torch.from_numpy(input_data))

    compare(outputs, nn.Softmax(dim=1)(pytorch_outputs))


def test_backward_pass(net_arguments):
    net, pytorch_net, layers = net_arguments
    input_data = np.array([[[[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 11, 12],
                             [13, 14, 15, 16]],
                            [[-1, -2, 0, 1],
                             [-2, -3, 1, 2],
                             [3, 4, 2, -1],
                             [-2, -3, -4, 9]]]],
                          dtype=np.float64)
    expected = np.array([2])

    def loss_grad(layer, b):
        gradients = []

        for i in range(3):
            result = layer.result.data[i]
            if i == expected[0]:
                result -= 1
            gradients.append(result)

        return gradients

    net.forward(input_data)
    net.backward(loss_grad)

    criterion = nn.CrossEntropyLoss()

    pytorch_net.zero_grad()
    outputs = pytorch_net(torch.from_numpy(input_data))
    loss = criterion(outputs, torch.from_numpy(expected))
    loss.backward()

    pytorch_layers = [pytorch_net.conv, pytorch_net.fc]
    devito_layers = [layers[1], layers[3]]

    for i in range(len(pytorch_layers)):
        pytorch_layer = pytorch_layers[i]
        devito_layer = devito_layers[i]

        compare(devito_layer.kernel_gradients.data,
                pytorch_layer.weight.grad)

        compare(devito_layer.bias_gradients.data,
                pytorch_layer.bias.grad)
