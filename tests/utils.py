import numpy as np
from os import environ


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
