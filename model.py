import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
from jax import random as jrandom

#from flax.optim import Adam, weight_norm
import optax
from optax import adamw
from flax import linen as nn
from flax.core import FrozenDict
from typing import Optional

import zlib
import os
import random
from typing import List, Dict

import numpy as np

class MultiHeadSelfAttention(nn.Module):
    d_model: int
    d_head: int
    nhead: int

    def setup(self):
        self.q_proj = nn.Dense(self.d_head * self.nhead)
        self.k_proj = nn.Dense(self.d_head * self.nhead)
        self.v_proj = nn.Dense(self.d_head * self.nhead)
        self.out_proj = nn.Dense(self.d_model)

    def __call__(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to separate heads
        q = q.reshape((q.shape[0], q.shape[1], self.nhead, self.d_head)).transpose((0, 2, 1, 3))
        k = k.reshape((k.shape[0], k.shape[1], self.nhead, self.d_head)).transpose((0, 2, 1, 3))
        v = v.reshape((v.shape[0], v.shape[1], self.nhead, self.d_head)).transpose((0, 2, 1, 3))

        attn_score = jnp.einsum('bhqd,bhkd->bhqk', q, k) / jnp.sqrt(self.d_head)
        attn_weights = nn.softmax(attn_score, axis=-1)
        attn_output = jnp.einsum('bhqk,bhvd->bhqd', attn_weights, v)

        # Concatenate heads and pass through the final projection layer
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape((x.shape[0], x.shape[1], self.d_head * self.nhead))
        return self.out_proj(attn_output)

class FeedForwardNetwork(nn.Module):
    d_model: int
    d_mlp: int

    def setup(self):
        self.layer1 = nn.Dense(self.d_mlp)
        self.layer2 = nn.Dense(self.d_model)

    def __call__(self, x):
        return self.layer2(nn.gelu(self.layer1(x)))

class DecoderLayer(nn.Module):
    d_model: int
    d_head: int
    d_mlp: int
    nhead: int
    layernorm: Optional[bool]

    def setup(self):
        self.self_attn = MultiHeadSelfAttention(self.d_model, self.d_head, self.nhead)
        self.ffn = FeedForwardNetwork(self.d_model, self.d_mlp)
        if self.layernorm:
            self.ln1 = nn.LayerNorm()
            self.ln2 = nn.LayerNorm()

    def __call__(self, x):
        attn_output = self.self_attn(x)
        if self.layernorm:
            x = self.ln1(x + attn_output)
        else:
            x = x + attn_output

        ffn_output = self.ffn(x)
        if self.layernorm:
            x = self.ln2(x + ffn_output)
        else:
            x = x + ffn_output

        return x

class TransformerDecoder(nn.Module):
    d_model: int
    d_head: int
    d_mlp: int
    nhead: int
    vocab_size: int
    n_layers: int
    layernorm: Optional[bool]

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.positional_embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)  # Assuming vocab size for positional as well
        self.decoder_layers = [DecoderLayer(self.d_model, self.d_head, self.d_mlp, self.nhead, self.layernorm) for _ in range(self.n_layers)]
        self.output_layer = nn.Dense(self.vocab_size)


    def __call__(self, x):
        x = self.embedding(x)
        pos = jnp.arange(0, x.shape[1])
        x += self.positional_embedding(pos)

        for layer in self.decoder_layers:
            x = layer(x)

        return self.output_layer(x)
    
class MLP(nn.Module):
    d_mlp: int
    d_model: int
    vocab_size: int
    n_layers: int
    fn_activation: str = "relu"

    def setup(self):
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.d_model)
        self.mlp_layers = [nn.Dense(self.d_mlp) for _ in range(self.n_layers)]
        self.output_layer = nn.Dense(self.vocab_size)
        if self.fn_activation == "relu":
            self.activation_fn = nn.relu
        else:
            self.activation_fn = nn.tanh

    def __call__(self, x):
        x = self.embedding(x)
        x = x.reshape((*x.shape[:-2], x.shape[-1] * x.shape[-2]))
        
        for i, layer in enumerate(self.mlp_layers):
            x = layer(x)
            if i != len(self.mlp_layers) - 1:
                x = self.activation_fn(x)

        return self.output_layer(x)[:, None, :]