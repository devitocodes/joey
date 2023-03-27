import math

from devito import Function, Operator, Eq, Inc, Constant, exp

from joey import Module, default_dim_allocator
from joey.utils import get_tensor_4d, get_tensor_3d
from joey.new_layers import FullyConnected3d

from torch import nn
import torch
from torch import functional as F


class MultiHeadAttention(Module):
    r"""Multi-headed Attention for input Query, Key, Value

        Multi-headed Attention is a module for attention mechanisms which runs through attention in several times in
        parallel, then the multiple outputs are concatenated and linearly transformed

        Args:
            embed_size  (int): Max embedding size
            num_heads   (int): Number of heads in multi-headed attention; Number of splits in the embedding size
            batch_dim   (int, optional): The dimension in which batch dimensions is

        """

    def __init__(self,
                 embed_size: int,
                 num_heads: int,
                 lines: int,
                 batch_size: int,
                 batch_dim: int = 0,
                 generate_code=False,
                 name='att'
                 ):
        self.name = name
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.batch_dim = batch_dim
        self.lines = lines
        self.batch_size = batch_size

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self._I = get_tensor_3d(name=('input_' + self.name),
                                shape=(self.batch_size, self.lines, self.embed_size))

        self.Q = FullyConnected3d(input_size=(self.batch_size, self.lines, self.embed_size),
                                  weight_size=(self.embed_size, self.embed_size))

        self.K = FullyConnected3d(input_size=(self.batch_size, self.lines, self.embed_size),
                                  weight_size=(self.embed_size, self.embed_size))

        self.V = FullyConnected3d(input_size=(self.batch_size, self.lines, self.embed_size),
                                  weight_size=(self.embed_size, self.embed_size))

        self.linear = FullyConnected3d(input_size=(self.batch_size, self.lines, self.embed_size),
                                       weight_size=(self.embed_size, self.embed_size))

        reshaped = (self.batch_size, self.lines, self.num_heads, self.head_size)
        shape_sum = (self.batch_size, self.num_heads, self.lines, 1)
        shape_scores = (self.batch_size, self.num_heads, self.lines, self.lines)

        self.q_reshaped = get_tensor_4d('q_4d_', reshaped)
        self.k_reshaped = get_tensor_4d('k_4d_', reshaped)
        self.v_reshaped = get_tensor_4d('v_4d_', reshaped)

        b, q, k, h, e, h1 = default_dim_allocator(6)

        self.sqrt_embeded = Constant(name + 'sqrt_embed', value=math.sqrt(self.embed_size))

        self.scores = get_tensor_4d(name=('bhqk' + self.name), shape=shape_scores, dims=[b, q, k, e])
        self.scores_result = get_tensor_4d(name=('scores_result' + self.name), shape=shape_scores)
        self.attention = get_tensor_4d(name=('attention' + self.name), shape=reshaped)
        self.expon = get_tensor_4d(name=('expon' + self.name), shape=shape_scores, dims=[b, q, k, e])
        self.sum_all = get_tensor_4d(name=('sum_all' + self.name), shape=shape_sum, dims=[b, q, k, h1])

        self._R = get_tensor_3d(name=('result_' + self.name), shape=(self.batch_size, self.lines, self.embed_size))

        if generate_code:
            eqs, args = self.equations()
            self._arg_dict = dict(args)
            self._op = Operator(eqs)
            self._op.cfunction

    def equations(self) -> (list, list):
        x1, y1, z1, w1 = self.Q._dimensions
        d1, d2, d3 = self.Q.input.dimensions

        q_a, q_b, q_c, q_d = self.q_reshaped.dimensions
        k_a, k_b, k_c, k_d = self.k_reshaped.dimensions
        v_a, v_b, v_c, v_d = self.v_reshaped.dimensions

        b, q, h, e = self.q_reshaped.dimensions
        _, k, _, _ = self.k_reshaped.dimensions

        b2, h2, q2, k2 = self.scores.dimensions
        b3, h3, q3, k3 = self.scores_result.dimensions
        _, _, _, h1 = self.sum_all.dimensions

        eqs = [
            Eq(self.Q.input[d1, d2, d3], self.input[d1, d2, d3]),
            Eq(self.K.input[d1, d2, d3], self.input[d1, d2, d3]),
            Eq(self.V.input[d1, d2, d3], self.input[d1, d2, d3]),
            *self.Q.equations(dims=(x1, y1, z1, w1))[0],
            *self.K.equations(dims=(x1, y1, z1, w1))[0],
            *self.V.equations(dims=(x1, y1, z1, w1))[0],
            # Forward Equations for Query Key and Value
            Eq(self.q_reshaped[q_a, q_b, q_c, q_d], self.Q.result[q_a, q_b, (q_c * self.head_size) + q_d]),
            Eq(self.k_reshaped[k_a, k_b, k_c, k_d], self.K.result[k_a, k_b, (k_c * self.head_size) + k_d]),
            Eq(self.v_reshaped[v_a, v_b, v_c, v_d], self.V.result[v_a, v_b, (v_c * self.head_size) + v_d]),
            # Einsum over Query and Key
            Eq(self.scores[b2, h2, q2, k2], 0),
            *[Inc(self.scores[b, i, q, k], self.q_reshaped[b, q, i, e] * self.k_reshaped[b, k, i, e]) for i in range(
                self.num_heads
            )],
            # Scores divided by sqrt(embed_size)
            Eq(self.scores[b2, h2, q2, k2], self.scores[b2, h2, q2, k2] / self.sqrt_embeded),
            # Sofmax(scores)
            Eq(self.expon[b2, h2, q2, k2], exp(self.scores[b2, h2, q2, k2])),
            Eq(self.sum_all[b2, h2, q2, h1], 0),
            Inc(self.sum_all[b2, h2, q2, h1], self.expon[b2, h2, q2, k2]),
            Eq(self.scores_result[b3, h3, q3, k3], self.expon[b3, h3, q3, k3] / self.sum_all[b3, h3, q3, h1]),
        ]

        i, k, j, l = self.attention.dimensions
        _, _, _, m = self.scores_result.dimensions
        a, b, c, d = self.attention.shape

        x, y, z = self._R.dimensions

        eqs += [
            Eq(self.attention[i, k, j, l], 0),
            *[Inc(self.attention[i, k, z, l], self.scores_result[i, z, k, m] * self.v_reshaped[i, m, z, l]) for z in
              range(self.num_heads)],
            Eq(self.linear.input[i, k, (j * d) + l], self.attention[i, k, j, l]),
            *self.linear.equations()[0],
            Eq(self.result[x, y, z], self.linear.result[x, y, z])
        ]

        return eqs, []

    def _allocate(self, **kwargs) -> (
            Function, Function, Function,
            Function, Function, Function,
            Function):
        pass

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []
