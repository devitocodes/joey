import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from devito import logger

from joey.models.ViT import ViT
from tests.utils import transfer_weights_ViT

logger.set_log_level(level='ERROR')
image_size = 28
channel_size = 1
patch_size = 7
embed_size = 512
num_heads = 8
classes = 10
num_layers = 3
hidden_size = 256
dropout = 0.2

np.random.seed(0)

BATCH_SIZE = 64

mean, std = (0.5,), (0.5,)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean, std)
                                ])

trainset = datasets.MNIST('../data/MNIST/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False)

testset = datasets.MNIST('../data/MNIST/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

model = ViT(image_size, channel_size, patch_size, embed_size, num_heads, classes, num_layers, hidden_size,
            generate_code=True)


def test_eval_model():
    transfer_weights_ViT(model)

    y_true_test = []
    y_pred_test = []

    for batch_idx, (img, labels) in enumerate(testloader):
        if img.size(0) != 64:
            continue
        preds = model.forward(img.detach().numpy())
        y_pred_test.extend(preds.argmax(axis=-1).tolist())
        y_true_test.extend(labels.detach().tolist())
        if batch_idx == 10:
            break

    total_correct = len([True for x, y in zip(y_pred_test, y_true_test) if x == y])
    total = len(y_pred_test)
    accuracy = total_correct * 100 / total

    print("Test Accuracy%: ", accuracy, "==", total_correct, "/", total)

    return accuracy >= 95

