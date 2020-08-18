import numpy as np
from os import environ


def compare(devito, pytorch):
    pytorch = pytorch.detach().numpy()

    if devito.shape != pytorch.shape:
        pytorch = np.transpose(pytorch)

    error = abs(devito - pytorch) / abs(pytorch)
    max_error = np.nanmax(error)

    if not np.isnan(max_error):
        assert(max_error < 1e-13)


def running_in_parallel():
    if 'DEVITO_LANGUAGE' not in environ:
        return False

    return environ['DEVITO_LANGUAGE'] in ['openmp']


def get_run_count():
    if running_in_parallel():
        return 1000
    else:
        return 1
