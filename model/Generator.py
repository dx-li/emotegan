import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from typing import List, Union
from jaxtyping import Array, Float, Int, PyTree
from model.Layers import SpectralConv2d
from functools import partial

def bilinear_upsample(x, scale_factor):
    new_shape = (x.shape[0], int(x.shape[1] * scale_factor), int(x.shape[2] * scale_factor))
    upsampled = jax.image.resize(x, new_shape, method='bilinear')
    return upsampled

class GResBlock(eqx.Module):
    skip_conv: SpectralConv2d
    conv_0: SpectralConv2d
    conv_1: SpectralConv2d
    # bn_1: eqx.nn.BatchNorm
    # bn_2: eqx.nn.BatchNorm
    up: partial

    def __init__(
            self,
            key,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            upscale_factor=2,
            dtype=jnp.bfloat16
    ):
        keys = jax.random.split(key, 3)
        self.conv_0 = SpectralConv2d(keys[0], in_channels, out_channels, kernel_size, stride=stride, padding=padding, dtype=dtype)
        self.conv_1 = SpectralConv2d(keys[1], out_channels, out_channels, kernel_size, stride=stride, padding=padding, dtype=dtype)
        self.skip_conv = SpectralConv2d(keys[2], in_channels, out_channels, 1, padding=0, dtype=dtype)
        # self.bn_1 = eqx.nn.BatchNorm(in_channels, axis_name="batch", dtype=dtype)
        # self.bn_2 = eqx.nn.BatchNorm(out_channels, axis_name="batch", dtype=dtype)
        self.up = partial(bilinear_upsample, scale_factor=upscale_factor)

    def __call__(self, x, state, key=None, inference=None):
        out = x
        # out, state = self.bn_1(out, state)
        out = jax.nn.relu(out)
        out = eqx.filter_vmap(self.up, in_axes=1, out_axes=1)(out)
        out, state = eqx.filter_vmap(self.conv_0, in_axes=(1, None), out_axes=(1, None))(out, state)
        # out, state = self.bn_2(out, state)
        out = jax.nn.relu(out)
        out, state = eqx.filter_vmap(self.conv_1, in_axes=(1, None), out_axes=(1, None))(out, state)
        skip = x
        skip = eqx.filter_vmap(self.up, in_axes=1, out_axes=1)(skip)
        skip, state = eqx.filter_vmap(self.skip_conv, in_axes=(1, None), out_axes=(1, None))(skip, state)
        return out + skip, state

class ConvGRUCell(eqx.Module):
    input_size: Int
    hidden_size: Int
    reset_gate: eqx.nn.Conv2d
    update_gate: eqx.nn.Conv2d
    out_gate: eqx.nn.Conv2d

    def __init__(
            self,
            key,
            input_size,
            hidden_size,
            kernel_size,
            dtype=jnp.bfloat16
    ):
        keys = jax.random.split(key, 3)
        padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reset_gate = eqx.nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding, use_bias=False, key=keys[0], dtype=dtype)
        self.update_gate = eqx.nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding, use_bias=False, key=keys[1], dtype=dtype)
        self.out_gate = eqx.nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding, use_bias=False, key=keys[2], dtype=dtype)

    def __call__(self, input, hidden, key=None):
        combined = jnp.concatenate([input, hidden], axis=0)
        reset = jax.nn.sigmoid(self.reset_gate(combined))
        update = jax.nn.sigmoid(self.update_gate(combined))
        combined = jnp.concatenate([input, hidden * reset], axis=0)
        out = jnp.tanh(self.out_gate(combined))
        new_hidden = hidden * (1 - update) + out * update
        return new_hidden

class InitialConvGRU(eqx.Module):
    conv_cell: ConvGRUCell
    n_frames: int
    hidden_size: int

    def __init__(self, key, input_size, hidden_size, kernel_size, n_frames, dtype=jnp.bfloat16):
        self.conv_cell = ConvGRUCell(key, input_size, hidden_size, kernel_size, dtype=dtype)
        self.n_frames = n_frames
        self.hidden_size = hidden_size

    def __call__(self, x):
        C, H, W = x.shape
        x = jnp.tile(x, (self.n_frames, 1, 1, 1))
        def process_frame(carry, frame):
            hidden_states = carry
            new_hidden_states = self.conv_cell(frame, hidden_states)
            return new_hidden_states, new_hidden_states

        initial_hidden = jnp.zeros((self.hidden_size, H, W), dtype=x.dtype)
        _, outputs = jax.lax.scan(process_frame, initial_hidden, x)
        return outputs

class TemporalConvGRU(eqx.Module):
    conv_cell: ConvGRUCell
    n_frames: int
    hidden_size: int

    def __init__(self, key, input_size, hidden_size, kernel_size, n_frames, dtype=jnp.bfloat16):
        self.conv_cell = ConvGRUCell(key, input_size, hidden_size, kernel_size, dtype=dtype)
        self.n_frames = n_frames
        self.hidden_size = hidden_size

    def __call__(self, x):
        T, C, H, W = x.shape
        assert T == self.n_frames, f"Expected {self.n_frames} frames, got {T}"

        def process_frame(carry, frame):
            hidden_states = carry
            new_hidden_states = self.conv_cell(frame, hidden_states)
            return new_hidden_states, new_hidden_states

        initial_hidden = jnp.zeros((self.hidden_size, H, W), dtype=x.dtype)
        _, outputs = jax.lax.scan(process_frame, initial_hidden, x)
        return outputs

class Generator(eqx.Module):
    in_dim: Int
    latent_dim: Int
    ch: Int
    n_frames: Int
    affine_transformation: eqx.nn.Linear
    conv_blocks: List[Union[GResBlock, TemporalConvGRU]]
    colorize: SpectralConv2d
    final: SpectralConv2d

    def __init__(
            self,
            key,
            in_dim,
            latent_dim,
            ch,
            n_frames=15,
            dtype=jnp.bfloat16
    ):
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.ch = ch
        self.n_frames = n_frames
        keys = jax.random.split(key, 15)
        self.affine_transformation = eqx.nn.Linear(in_dim, latent_dim * latent_dim * 8 * ch, key=keys[0], dtype=dtype)
        self.conv_blocks = [
            InitialConvGRU(keys[1], 8 * ch, 8 * ch, 5, n_frames, dtype=dtype),
            GResBlock(keys[2], 8 * ch, 8 * ch, upscale_factor=1, dtype=dtype),
            GResBlock(keys[3], 8 * ch, 8 * ch, dtype=dtype),
            TemporalConvGRU(keys[4], 8 * ch, 8 * ch, 5, n_frames, dtype=dtype),
            GResBlock(keys[5], 8 * ch, 8 * ch, upscale_factor=1, dtype=dtype),
            GResBlock(keys[6], 8 * ch, 8 * ch, dtype=dtype),
            # TemporalConvGRU(keys[7], 8 * ch, 8 * ch, 5, n_frames, dtype=dtype),
            GResBlock(keys[8], 8 * ch, 8 * ch, upscale_factor=1, dtype=dtype),
            GResBlock(keys[9], 8 * ch, 8 * ch, dtype=dtype),
            TemporalConvGRU(keys[10], 8 * ch, 4 * ch, 5, n_frames, dtype=dtype),
            GResBlock(keys[11], 4 * ch, 4 * ch, upscale_factor=1, dtype=dtype),
            GResBlock(keys[12], 4 * ch, 2 * ch, dtype=dtype),
        ]

        self.colorize = SpectralConv2d(keys[13], 2 * ch, 3, 3, padding=1, dtype=dtype)
        self.final = SpectralConv2d(keys[14], 3, 3, 9, padding=0, dtype=dtype)

    def __call__(self, x, state, key=None):
        y = self.affine_transformation(x)
        y = jnp.reshape(y, (8 * self.ch, self.latent_dim, self.latent_dim))
        for conv in self.conv_blocks:
            if isinstance(conv, InitialConvGRU):
                y = conv(y)
                y = jnp.transpose(y, (1, 0, 2, 3))
            elif isinstance(conv, TemporalConvGRU):
                y = jnp.transpose(y, (1, 0, 2, 3))
                y = conv(y)
                y = jnp.transpose(y, (1, 0, 2, 3))
            elif isinstance(conv, GResBlock):
                y, state = conv(y, state)
        y = jax.nn.relu(y)
        y, state = eqx.filter_vmap(self.colorize, in_axes=(1, None), out_axes=(1, None))(y, state)
        y = jax.nn.relu(y)
        y, state = eqx.filter_vmap(self.final, in_axes=(1, None), out_axes=(1, None))(y, state)
        
        return y, state

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    subkey1, subkey2 = jax.random.split(subkey)
    x1 = jax.random.normal(subkey1, (10, 120), dtype=jnp.bfloat16)
    x2 = jax.random.normal(subkey2, (10, 120), dtype=jnp.bfloat16)
    model, state = eqx.nn.make_with_state(Generator)(key, 120, 4, 32)
    model_vmap = eqx.filter_vmap(model, axis_name='batch', in_axes=(0, None), out_axes=(0, None))
    model_jit = eqx.filter_jit(model_vmap)
    y, state = model_jit(x1, state)
    print(y)
    print(y.shape)