import numpy as np
import torch
from devito import Function, Operator, Eq, Inc
from torch import nn

from joey import Module
from joey.activation import ReLU
from joey.module.MultiHeadAttention import MultiHeadAttention, MultiHeadAttentionTorch
from joey.new_layers import Norm3d, FullyConnected3d
from joey.utils import get_tensor_3d


class VisionEncoder(Module):
    r"""Vision Encoder Model

           An Encoder Layer with the added functionality to encode important local structures of a tokenized image

           Args:
               embed_size      (int): Embedding Size of Input
               num_heads       (int): Number of heads in multi-headed attention
               hidden_size     (int): Number of hidden layers
               dropout         (float, optional): A probability from 0 to 1 which determines the dropout rate

       """
    def __init__(self,
                 embed_size: int,
                 num_heads: int,
                 hidden_size: int,
                 lines: int,
                 batch_size: int,
                 dropout: float = 0.1,
                 name='vision_encoder',
                 generate_code=False
                 ):
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.name = name
        self.batch_size = batch_size
        self.lines = lines

        self.norm1 = Norm3d(input_size=(self.batch_size, self.lines, self.embed_size),
                            weight_size=(self.embed_size,),
                            name='norm_1_' + self.name)
        self.norm2 = Norm3d(input_size=(self.batch_size, self.lines, self.embed_size),
                            weight_size=(self.embed_size,),
                            name='norm_2_' + self.name)

        self._I = self.norm1.input

        self.attention = MultiHeadAttention(
            embed_size=self.embed_size,
            batch_dim=0,
            num_heads=self.num_heads,
            lines=self.lines,
            batch_size=self.batch_size,
            generate_code=False,
            name=name + '_att_'
        )

        first_in = (self.batch_size, self.lines, self.embed_size)
        second_in = (self.batch_size, self.lines, self.embed_size * 4)

        mlp1 = FullyConnected3d(input_size=first_in,
                                weight_size=(4 * self.embed_size, self.embed_size),
                                activation=ReLU())
        mlp2 = FullyConnected3d(input_size=second_in,
                                weight_size=(self.embed_size, 4 * self.embed_size,))
        self.mlp = [mlp1, mlp2]

        self._R = get_tensor_3d('result_encoder_' + self.name, (self.batch_size, self.lines, self.embed_size))

        if generate_code:
            eqs, args = self.equations()
            self._arg_dict = dict(args)
            self._op = Operator(eqs)
            self._op.cfunction

    def equations(self) -> (list, list):
        a, b, c = self.result.dimensions
        x, y, z = self.attention.result.dimensions
        u, v, t = self.mlp[0].input.dimensions
        g, j, k = self.mlp[1].input.dimensions
        p, q, r = self.result.dimensions

        return [
            Eq(self.result[a, b, c], 0),
            *self.norm1.equations()[0],
            Eq(self.result[a, b, c], self.norm1.result[a, b, c]),
            Eq(self.attention.input[a, b, c], self.result[a, b, c]),
            *self.attention.equations()[0],
            Inc(self.result[x, y, z], self.attention.result[x, y, z]),
            Eq(self.norm2.input[x, y, z], self.result[x, y, z]),
            *self.norm2.equations()[0],
            Eq(self.mlp[0].input[u, v, t], self.norm2.result[u, v, t]),
            *self.mlp[0].equations()[0],
            Eq(self.mlp[1].input[g, j, k], self.mlp[0].result[g, j, k]),
            *self.mlp[1].equations()[0],
            Inc(self.result[p, q, r], self.mlp[1].result[p, q, r])
        ], []

    def _allocate(self, **kwargs) -> (Function, Function, Function,
                                      Function, Function, Function,
                                      Function):
        pass

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        pass

