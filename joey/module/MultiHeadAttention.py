import math

import numpy as np

from devito import Function, Operator, Eq, Inc, Constant, exp

from joey import Module, default_name_allocator
from joey.utils import get_tensor_4d, get_tensor_3d
from joey.new_layers import FullyConnected3d
from joey.funtional import Softmax4d, Expand3to4, Contract4to3

import time
from torch import nn
import torch
from torch import functional as F


class MultiHeadAttention(Module):

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

        self._R = get_tensor_3d(name=('result_' + self.name), shape=(self.batch_size, self.lines, self.embed_size))

        if generate_code:
            eqs, args = self.equations()
            self._arg_dict = dict(args)
            self._op = Operator(eqs)
            self._op.cfunction

    def equations(self) -> (list, list):
        x1, y1, z1, w1 = self.Q._dimensions

        reshaped = (self.batch_size, self.lines, self.num_heads, self.head_size)
        shape_scores = (self.batch_size, self.num_heads, self.lines, self.lines)

        scores = get_tensor_4d(name=('bhqk_' + self.name), shape=shape_scores)
        scores_result = get_tensor_4d(name=('scores_result_' + self.name), shape=shape_scores)
        attention = get_tensor_4d(name=('attention_' + self.name), shape=reshaped)

        sqrt_embed = Constant(name=('sqrt_embed_' + self.name), value=math.sqrt(self.embed_size))

        q_reshaped = Expand3to4(name=('q_reshaped_' + self.name), shape_in=self.Q.result.shape, shape_out=reshaped)
        k_reshaped = Expand3to4(name=('k_reshaped_' + self.name), shape_in=self.K.result.shape, shape_out=reshaped)
        v_reshaped = Expand3to4(name=('v_reshaped_' + self.name), shape_in=self.V.result.shape, shape_out=reshaped)
        soft_score = Softmax4d(name='soft_' + self.name, shape=shape_scores)

        q_a, q_b, q_c = q_reshaped.input.dimensions
        k_a, k_b, k_c = k_reshaped.input.dimensions
        v_a, v_b, v_c = v_reshaped.input.dimensions

        b, q, h, e = q_reshaped.dimensions
        _, k, _, _ = k_reshaped.dimensions

        b2, h2, q2, k2 = scores.dimensions
        b3, h3, q3, k3 = soft_score.dimensions

        eqs = [
            *self.Q.equations()[0],
            *self.K.equations()[0],
            *self.V.equations()[0],
            Eq(q_reshaped.input[q_a, q_b, q_c], self.Q.result[q_a, q_b, q_c]),
            Eq(k_reshaped.input[k_a, k_b, k_c], self.K.result[k_a, k_b, k_c]),
            Eq(v_reshaped.input[v_a, v_b, v_c], self.V.result[v_a, v_b, v_c]),
            *q_reshaped.equations()[0],
            *k_reshaped.equations()[0],
            *v_reshaped.equations()[0],
            Eq(scores[b2, h2, q2, k2], 0),
            *[Inc(scores[b, i, q, k], q_reshaped.result[b, q, i, e] * k_reshaped.result[b, k, i, e]) for i in
              range(self.num_heads)
              ],
            Eq(scores[b2, h2, q2, k2], scores[b2, h2, q2, k2] / sqrt_embed),
            Eq(soft_score.input[b3, h3, q3, k3], scores[b3, h3, q3, k3]),
            *soft_score.equations()[0]
        ]

        i, k, j, l = attention.dimensions
        _, _, _, m = soft_score.dimensions
        _, _, _, d = attention.shape

        reductor = Contract4to3(name='contract_' + self.name, shape_in=attention.shape,
                                shape_out=self.linear.result.shape)

        a, b, c, d = reductor.input.dimensions
        e, f, g = self.linear.input.dimensions
        x, y, z = self._R.dimensions

        eqs += [
            Eq(attention[i, k, j, l], 0),
            *[Inc(attention[i, k, z, l], soft_score.result[i, z, k, m] * v_reshaped.result[i, m, z, l]) for z in
              range(self.num_heads)],
            Eq(reductor.input[a, b, c, d], attention[a, b, c, d]),
            *reductor.equations()[0],
            Eq(self.linear.input[e, f, g], reductor.result[e, f, g]),
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


class MultiHeadAttentionTorch(nn.Module):
    r"""Multi-headed Attention for input Query, Key, Value

    Multi-headed Attention is a module for attention mechanisms which runs through attention in several times in
    parallel, then the multiple outputs are concatenated and linearly transformed

    Args:
        embed_size  (int): Max embedding size
        num_heads   (int): Number of heads in multi-headed attention; Number of splits in the embedding size
        dropout     (float, optional): Percentage of Dropout to be applied in range 0 <= dropout <=1
        batch_dim   (int, optional): The dimension in which batch dimensions is

    """

    def __init__(self, embed_size: int, num_heads: int, dropout: float = 0.2, batch_dim: int = 0):
        super(MultiHeadAttentionTorch, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_dim = batch_dim

        self.dropout_layer = nn.Dropout(dropout)

        self.head_size = self.embed_size // self.num_heads

        assert self.head_size * self.num_heads == self.embed_size, "Heads cannot split Embedding size equally"

        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)

        self.linear = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, q, k, v, mask=None):
        out = self.batch_0(q, k, v, mask)

        return out

    def batch_0(self, q, k, v, mask=None):
        q_batch_size, q_seq_len, q_embed_size = q.size()
        k_batch_size, k_seq_len, k_embed_size = k.size()
        v_batch_size, v_seq_len, v_embed_size = v.size()

        q = self.Q(q).reshape(q_batch_size, q_seq_len, self.num_heads, self.head_size)
        k = self.K(k).reshape(k_batch_size, k_seq_len, self.num_heads, self.head_size)
        v = self.V(v).reshape(v_batch_size, v_seq_len, self.num_heads, self.head_size)

        self.q_reshaped = q
        self.k_reshaped = k
        self.v_reshaped = v

        attention = self.attention(q, k, v, mask=mask)
        self.att = attention
        concatenated = attention.reshape(v_batch_size, -1, self.embed_size)
        self.concatened = concatenated
        out = self.linear(concatenated)

        return out

    def attention(self, q, k, v, mask=None):
        scores = torch.einsum("bqhe,bkhe->bhqk", [q, k])

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        scores /= math.sqrt(self.embed_size)
        scores = F.F.softmax(scores, dim=-1)
        attention = torch.einsum("bhql,blhd->bqhd", [scores, v])
        return attention

a = MultiHeadAttention(
    embed_size=512,
    batch_dim=0,
    num_heads=8,
    lines=17,
    batch_size=64,
    generate_code=True
)

# print(a.eqs)
print(a._op)
# a.operator1.apply()

b = MultiHeadAttentionTorch(
    embed_size=512,
    dropout=0.2,
    batch_dim=0,
    num_heads=8
)

a.Q.kernel.data[:] = b.Q.weight.detach().numpy()
a.Q.bias.data[:] = b.Q.bias.detach().numpy()
a.V.kernel.data[:] = b.V.weight.detach().numpy()
a.V.bias.data[:] = b.V.bias.detach().numpy()
a.K.kernel.data[:] = b.K.weight.detach().numpy()
a.K.bias.data[:] = b.K.bias.detach().numpy()

a.linear.kernel.data[:] = b.linear.weight.detach().numpy()
a.linear.bias.data[:] = b.linear.bias.detach().numpy()
#
q = np.random.rand(64, 17, 512)
k = np.random.rand(64, 17, 512)
v = np.random.rand(64, 17, 512)
#
q = q.astype('float32')
k = k.astype('float32')
v = v.astype('float32')


a.Q.input.data[:] = q
a.K.input.data[:] = k
a.V.input.data[:] = v


t = time.time()
a._op.apply()
print(time.time() - t)
result = a.result.data

t2 = time.time()
result2 = b.forward(torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v))
print(time.time() - t2)

#
close = np.allclose(result, result2.detach().numpy(), rtol=1e-5, atol=1e-5)
diff = result2.detach().numpy() - result
print("pertim:", close)
