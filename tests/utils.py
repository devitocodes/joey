import numpy as np
from os import environ

import torch


def compare(devito, pytorch, tolerance):
    pytorch = pytorch.detach().numpy()

    if devito.shape != pytorch.shape:
        pytorch = np.transpose(pytorch)

    error = abs(devito - pytorch) / abs(pytorch)
    max_error = np.nanmax(error)

    assert(np.isnan(max_error) or max_error < tolerance)


def running_in_parallel():
    if 'DEVITO_LANGUAGE' not in environ:
        return False

    return environ['DEVITO_LANGUAGE'] in ['openmp']


def get_run_count():
    if running_in_parallel():
        return 1000
    else:
        return 1


def transfer_weights_ViT(model):

    weights_pretained = torch.load('../examples/resources/model_weights_ViT')

    def equal_layer(num):
        model.encoders[num].norm1.kernel.data[:] = weights_pretained[f'encoders.{num}.norm1.weight'].detach().numpy()
        model.encoders[num].norm1.bias.data[:] = weights_pretained[f'encoders.{num}.norm1.bias'].detach().numpy()
        model.encoders[num].norm2.kernel.data[:] = weights_pretained[f'encoders.{num}.norm2.weight'].detach().numpy()
        model.encoders[num].norm2.bias.data[:] = weights_pretained[f'encoders.{num}.norm2.bias'].detach().numpy()
        model.encoders[num].attention.Q.kernel.data[:] = weights_pretained[
            f'encoders.{num}.attention.Q.weight'].detach().numpy()
        model.encoders[num].attention.Q.bias.data[:] = weights_pretained[f'encoders.{num}.attention.Q.bias'].detach().numpy()
        model.encoders[num].attention.K.kernel.data[:] = weights_pretained[
            f'encoders.{num}.attention.K.weight'].detach().numpy()
        model.encoders[num].attention.K.bias.data[:] = weights_pretained[f'encoders.{num}.attention.K.bias'].detach().numpy()
        model.encoders[num].attention.V.kernel.data[:] = weights_pretained[
            f'encoders.{num}.attention.V.weight'].detach().numpy()
        model.encoders[num].attention.V.bias.data[:] = weights_pretained[f'encoders.{num}.attention.V.bias'].detach().numpy()
        model.encoders[num].attention.linear.kernel.data[:] = weights_pretained[
            f'encoders.{num}.attention.linear.weight'].detach().numpy()
        model.encoders[num].attention.linear.bias.data[:] = weights_pretained[
            f'encoders.{num}.attention.linear.bias'].detach().numpy()
        model.encoders[num].mlp[0].kernel.data[:] = weights_pretained[f'encoders.{num}.mlp.0.weight'].detach().numpy()
        model.encoders[num].mlp[0].bias.data[:] = weights_pretained[f'encoders.{num}.mlp.0.bias'].detach().numpy()
        model.encoders[num].mlp[1].kernel.data[:] = weights_pretained[f'encoders.{num}.mlp.2.weight'].detach().numpy()
        model.encoders[num].mlp[1].bias.data[:] = weights_pretained[f'encoders.{num}.mlp.2.bias'].detach().numpy()

    with torch.no_grad():
        model.embeddings.kernel.data[:] = weights_pretained['embeddings.weight'].detach().numpy()
        model.embeddings.bias.data[:] = weights_pretained['embeddings.bias'].detach().numpy()
        model.class_token.data[:] = weights_pretained['class_token'].detach().numpy()
        model.positional_encoding.data[:] = weights_pretained['positional_encoding'].detach().numpy()

        for i in range(len(model.encoders)):
            equal_layer(i)

        model.norm.kernel.data[:] = weights_pretained['norm.weight'].detach().numpy()
        model.norm.bias.data[:] = weights_pretained['norm.bias'].detach().numpy()
        model.classifier.kernel.data[:] = weights_pretained['classifier.0.weight'].detach().numpy()
        model.classifier.bias.data[:] = weights_pretained['classifier.0.bias'].detach().numpy()


