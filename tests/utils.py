import numpy as np


def compare(devito, pytorch):
    pytorch = pytorch.detach().numpy()

    if devito.shape != pytorch.shape:
        pytorch = np.transpose(pytorch)

    error = abs(devito - pytorch) / abs(pytorch)
    max_error = np.nanmax(error)

    if max_error != np.nan:
        assert(max_error < 10**(-9))
