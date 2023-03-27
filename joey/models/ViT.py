from joey import Module, default_dim_allocator
from joey.module.VisionEncoder import VisionEncoder
from joey.utils import get_tensor_3d
from joey.new_layers import FullyConnected3d, Norm2d, FullyConnected2d
from devito import Operator, Inc, Eq, Function
from scipy.special import log_softmax
import numpy as np


class ViT(Module):
    r"""Vision Transformer Model

        A transformer model to solve vision tasks by treating images as sequences of tokens.

        Args:
            image_size      (int): Size of input image
            channel_size    (int): Size of the channel
            patch_size      (int): Max patch size, determines number of split images/patches and token size
            embed_size      (int): Embedding size of input
            num_heads       (int): Number of heads in Multi-Headed Attention
            classes         (int): Number of classes for classification of data
            hidden_size     (int): Number of hidden layers

    """

    def __init__(self, image_size: int, channel_size: int, patch_size: int, embed_size: int, num_heads: int,
                 classes: int, num_layers: int, hidden_size: int, batch: int = 64, generate_code=False):

        self.p = patch_size
        self.image_size = image_size
        self.embed_size = embed_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = channel_size * (patch_size ** 2)
        self.num_heads = num_heads
        self.classes = classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        img_shape = (batch, int((self.image_size / self.p) * (self.image_size / self.p)), self.patch_size)

        self._R = get_tensor_3d('result_1srt', (batch, self.num_patches + 1, self.embed_size))

        d, e = default_dim_allocator(2)
        x, y, z = self._R.dimensions

        self.embeddings = FullyConnected3d(input_size=img_shape, weight_size=(self.embed_size, self.patch_size))
        self.class_token = get_tensor_3d('class_token', (1, 1, self.embed_size), dims=(d, e, z))
        self.positional_encoding = get_tensor_3d('pos_enc', (1, self.num_patches + 1, self.embed_size), dims=(d, y, z))

        self.class_token.data[:] = np.random.rand(*self.class_token.shape)
        self.positional_encoding.data[:] = np.random.rand(*self.positional_encoding.shape)

        self.encoders = []
        for layer in range(self.num_layers):
            vision_encoder = VisionEncoder(
                embed_size=self.embed_size,
                num_heads=self.num_heads,
                batch_size=batch,
                lines=self.num_patches + 1,
                hidden_size=self.hidden_size,
                name='encoder' + str(layer)
            )
            self.encoders.append(vision_encoder)

        self.norm = Norm2d(input_size=(batch, self.embed_size), weight_size=(self.embed_size,))
        self.classifier = FullyConnected2d(input_size=(batch, self.embed_size), weight_size=(self.classes,
                                                                                             self.embed_size))

        if generate_code:
            eqs, args = self.equations()
            self._arg_dict = dict(args)
            self._op = Operator(eqs)
            self._op.cfunction

    def equations(self):

        a, b, c = self.embeddings.result.dimensions
        d, e, _ = self.class_token.dimensions
        x, y, z = self.result.dimensions

        t0, u0, v0 = self.encoders[0].norm1.input.dimensions

        eqs = [
            Eq(self.result, 0),
            *self.embeddings.equations()[0],
            Eq(self.result[a, b, c], self.embeddings.result[a, b, c]),
            Eq(self.result[x, self.num_patches, z], self.class_token[0, 0, z]),
            Inc(self.result[x, y, z], self.positional_encoding[d, y, z]),
            Eq(self.encoders[0].norm1.input[t0, u0, v0], self.result[t0, u0, v0])
        ]

        for index, encoder in enumerate(self.encoders):
            if index > 0:
                t, u, v = self.encoders[index].input.dimensions
                eqs.append(
                    Eq(self.encoders[index].input[t, u, v], self.encoders[index-1].result[t, u, v])
                )
            eqs += encoder.equations()[0]

        last_enconder = self.encoders[-1].result

        a, b = self.norm.input.dimensions
        i, j = self.classifier.input.dimensions
        _, x, _ = last_enconder.shape
        eqs += [
            Eq(self.norm.input[a, b], last_enconder[a, x - 1, b]),
            *self.norm.equations()[0],
            Eq(self.classifier.input[i, j], self.norm.result[i, j]),
            *self.classifier.equations()[0]
        ]

        return eqs, []

    def _allocate(self, **kwargs) -> (Function, Function, Function,
                                      Function, Function, Function,
                                      Function):
        pass

    def backprop_equations(self, prev_layer, next_layer) -> (list, list):
        return [], []

    def forward(self, x):

        b, c, h, w = x.shape
        x = x.reshape(b, int((h / self.p) * (w / self.p)), c * self.p * self.p)
        self.embeddings.input.data[:] = x

        self._op.apply()

        return log_softmax(self.classifier.result.data, axis=-1)