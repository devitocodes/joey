from devito import Function, SpaceDimension
import numpy as np

from joey import default_dim_allocator, default_name_allocator


def get_tensor_4d(name, shape, dims=None):
    a, b, c, d = shape
    _a, _b, _c, _d = default_dim_allocator(4) if not dims else dims

    return Function(name=default_name_allocator(name), shape=(a, b, c, d), dimensions=(_a, _b, _c, _d),
                    dtype=np.float32)


def get_tensor_3d(name, shape, dims=None):
    a, b, c = shape
    _a, _b, _c = default_dim_allocator(3) if not dims else dims

    return Function(name=default_name_allocator(name), shape=(a, b, c), dimensions=(_a, _b, _c), dtype=np.float32)


def get_tensor_2d(name, shape, dims=None):
    a, b = shape
    _a, _b = default_dim_allocator(2) if not dims else dims

    return Function(name=default_name_allocator(name), shape=(a, b), dimensions=(_a, _b), dtype=np.float32)


def get_tensor_1d(name, shape, dim=None):
    a = shape
    _a = default_dim_allocator(1)[0] if not dim else dim

    return Function(name=default_name_allocator(name), shape=(a,), dimensions=(_a,), dtype=np.float32)
