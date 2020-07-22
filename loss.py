from math import log


def cross_entropy(expected, actual):
    return -sum([expected[i]*log(actual[i]) for i in range(len(expected))])
