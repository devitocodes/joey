import pytest
import joey
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# PyTorch class
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Helper functions
SEED = 282757891
BATCH_SIZE = 4


@pytest.fixture
def net_arguments():
    np.random.seed(SEED)

    # Six 3x3 filters, activation RELU
    layer1 = joey.Conv(kernel_size=(6, 3, 3),
                       input_size=(BATCH_SIZE, 1, 32, 32),
                       activation=joey.activation.ReLU(),
                       generate_code=False)
    # 2x2 max pooling
    layer2 = joey.MaxPooling(kernel_size=(2, 2),
                             input_size=(BATCH_SIZE, 6, 30, 30),
                             stride=(2, 2),
                             generate_code=False)
    # Sixteen 3x3 filters, activation RELU
    layer3 = joey.Conv(kernel_size=(16, 3, 3),
                       input_size=(BATCH_SIZE, 6, 15, 15),
                       activation=joey.activation.ReLU(),
                       generate_code=False)
    # 2x2 max pooling
    layer4 = joey.MaxPooling(kernel_size=(2, 2),
                             input_size=(BATCH_SIZE, 16, 13, 13),
                             stride=(2, 2),
                             strict_stride_check=False,
                             generate_code=False)
    # Full connection (16 * 6 * 6 -> 120), activation RELU
    layer5 = joey.FullyConnected(weight_size=(120, 576),
                                 input_size=(576, BATCH_SIZE),
                                 activation=joey.activation.ReLU(),
                                 generate_code=False)
    # Full connection (120 -> 84), activation RELU
    layer6 = joey.FullyConnected(weight_size=(84, 120),
                                 input_size=(120, BATCH_SIZE),
                                 activation=joey.activation.ReLU(),
                                 generate_code=False)
    # Full connection (84 -> 10), output layer
    layer7 = joey.FullyConnectedSoftmax(weight_size=(10, 84),
                                        input_size=(84, BATCH_SIZE),
                                        generate_code=False)
    # Flattening layer necessary between layer 4 and 5
    layer_flat = joey.Flat(input_size=(BATCH_SIZE, 16, 6, 6),
                           generate_code=False)

    layers = [layer1, layer2, layer3, layer4,
              layer_flat, layer5, layer6, layer7]

    net = joey.Net(layers)

    pytorch_net = Net()
    pytorch_net.double()

    with torch.no_grad():
        pytorch_net.conv1.weight[:] = torch.from_numpy(layer1.kernel.data)
        pytorch_net.conv1.bias[:] = torch.from_numpy(layer1.bias.data)

        pytorch_net.conv2.weight[:] = torch.from_numpy(layer3.kernel.data)
        pytorch_net.conv2.bias[:] = torch.from_numpy(layer3.bias.data)

        pytorch_net.fc1.weight[:] = torch.from_numpy(layer5.kernel.data)
        pytorch_net.fc1.bias[:] = torch.from_numpy(layer5.bias.data)

        pytorch_net.fc2.weight[:] = torch.from_numpy(layer6.kernel.data)
        pytorch_net.fc2.bias[:] = torch.from_numpy(layer6.bias.data)

        pytorch_net.fc3.weight[:] = torch.from_numpy(layer7.kernel.data)
        pytorch_net.fc3.bias[:] = torch.from_numpy(layer7.bias.data)

    return (net, pytorch_net, layers)


@pytest.fixture(scope='module')
def mnist(tmpdir_factory):
    path = tmpdir_factory.mktemp('mnist')

    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize(0.5, 0.5)])
    trainset = torchvision.datasets.MNIST(root=path,
                                          train=True, download=True,
                                          transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=2)
    testset = torchvision.datasets.MNIST(root=path,
                                         train=False, download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    return (trainloader, testloader)


def compare(devito, pytorch):
    pytorch = pytorch.detach().numpy()

    if devito.shape != pytorch.shape:
        pytorch = np.transpose(pytorch)

    error = abs(devito - pytorch) / abs(pytorch)
    max_error = np.nanmax(error)

    if max_error != np.nan:
        assert(max_error < 10**(-9))


# Proper test functions
def test_forward_pass(net_arguments, mnist):
    _, mnist_test = mnist

    net, pytorch_net, layers = net_arguments
    images, _ = iter(mnist_test).next()

    images = images.double()

    outputs = net.forward(images.numpy())
    pytorch_outputs = pytorch_net(images)

    compare(outputs, nn.Softmax(dim=1)(pytorch_outputs))


def test_backward_pass(net_arguments, mnist):
    mnist_train, _ = mnist

    net, pytorch_net, layers = net_arguments

    images, labels = iter(mnist_train).next()

    def loss_grad(layer, b):
        gradients = []

        for j in range(10):
            result = layer.result.data[j, b]
            if j == labels[b]:
                result -= 1
            gradients.append(result)

        return gradients

    images = images.double()

    net.forward(images.numpy())
    net.backward(loss_grad)

    criterion = nn.CrossEntropyLoss()

    pytorch_net.zero_grad()
    outputs = pytorch_net(images)
    loss = criterion(outputs, labels)
    loss.backward()

    pytorch_layers = [pytorch_net.conv1, pytorch_net.conv2,
                      pytorch_net.fc1, pytorch_net.fc2, pytorch_net.fc3]
    devito_layers = [layers[0], layers[2], layers[5], layers[6], layers[7]]

    for i in range(len(pytorch_layers)):
        pytorch_layer = pytorch_layers[i]
        devito_layer = devito_layers[i]

        compare(devito_layer.kernel_gradients.data,
                pytorch_layer.weight.grad)

        compare(devito_layer.bias_gradients.data,
                pytorch_layer.bias.grad)


def test_training_sgd(net_arguments, mnist):
    pass
