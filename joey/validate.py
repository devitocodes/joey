import numpy as np
from torch import optim
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from devito import logger
import torch

from new_layers import x as net

# logger.set_log_level(level='ERROR')

mean, std = (0.5,), (0.5,)
BATCH_SIZE = 64

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])

trainset = datasets.MNIST('../data/MNIST/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)

testset = datasets.MNIST('../data/MNIST/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.SGD(net.pytorch_parameters, lr=0.001, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()

def loss_grad(result, expected):
    gradients = []
    for b in range(len(result.result.data)):
        row = []
        for i in range(10):
            result = [i, b]
            if i == expected[b]:
                result -= 1
            row.append(result)
        gradients.append(row)

    return gradients


for img, label in trainloader:
    img = img.reshape(28, 28, 64)

    # print("Input Image Dimensions: {}".format(img.size()))
    # print("Label Dimensions: {}".format(label.size()))
    # print("-" * 100)

    out = net.forward(img.detach().numpy())
    # loss = criterion(torch.from_numpy(out), label.t())
    # print(loss)
    net.backward(np.random.rand(64, 10), loss_grad, optimizer)

    # print("Output Dimensions: {}".format(out.shape))
    # break


# sys.exit(0)
