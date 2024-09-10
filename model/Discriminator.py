import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int, PyTree
from model.Layers import SpectralConv2d

class SpectralConv3d(eqx.nn.StatefulLayer):
    spec_conv: eqx.nn.SpectralNorm[eqx.nn.Conv3d]

    def __init__(
            self,
            key,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            dtype=jnp.bfloat16,
    ):
        key1, key2 = jax.random.split(key)
        self.spec_conv = eqx.nn.SpectralNorm(
            eqx.nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, key=key1, dtype=dtype),
            weight_name="weight",
            key=key2
        )

    def __call__(self, x, state, key=None, inference=None):
        return self.spec_conv(x, state)

class DResBlock(eqx.nn.StatefulLayer):
    skip_conv: SpectralConv2d
    conv_0: SpectralConv2d
    conv_1: SpectralConv2d
    
    def __init__(
            self,
            key,
            in_channels,
            out_channels,
            dtype=jnp.bfloat16,
    ):
        keys = jax.random.split(key, 3)
        self.skip_conv = SpectralConv2d(keys[0], in_channels, out_channels, 1, padding=0, dtype=dtype)
        self.conv_0 = SpectralConv2d(keys[1], in_channels, out_channels, 3, dtype=dtype)
        self.conv_1 = SpectralConv2d(keys[2], out_channels, out_channels, 3, dtype=dtype)

    def __call__(self, x, state, key=None, inference=None):
        out = x
        out = jax.nn.relu(out)
        out, state = self.conv_0(out, state)
        out = jax.nn.relu(out)
        out, state = self.conv_1(out, state)
        out = eqx.nn.AvgPool2d(2)(out)
        skip = x
        skip, state = self.skip_conv(skip, state)
        skip = eqx.nn.AvgPool2d(2)(skip)
        return out + skip, state

class D3DResBlock(eqx.nn.StatefulLayer):
    skip_conv: SpectralConv3d
    conv_0: SpectralConv3d
    conv_1: SpectralConv3d
    
    def __init__(
            self,
            key,
            in_channels,
            out_channels,
            dtype=jnp.bfloat16,
    ):
        keys = jax.random.split(key, 3)
        self.skip_conv = SpectralConv3d(keys[0], in_channels, out_channels, 1, padding=0, dtype=dtype)
        self.conv_0 = SpectralConv3d(keys[1], in_channels, out_channels, 3, dtype=dtype)
        self.conv_1 = SpectralConv3d(keys[2], out_channels, out_channels, 3, dtype=dtype)

    def __call__(self, x, state, key=None, inference=None):
        out = x
        out = jax.nn.relu(out)
        out, state = self.conv_0(out, state)
        out = jax.nn.relu(out)
        out, state = self.conv_1(out, state)
        out = eqx.nn.AvgPool3d(2)(out)
        skip = x
        skip, state = self.skip_conv(skip, state)
        skip = eqx.nn.AvgPool3d(2)(skip)
        return out + skip, state

class SpatialDiscriminator(eqx.Module):
    pre_conv: eqx.nn.Sequential
    pre_skip: SpectralConv2d
    conv_1: DResBlock
    conv_2: eqx.nn.Sequential
    linear: eqx.nn.SpectralNorm[eqx.nn.Linear]

    def __init__(
            self, 
            key,
            chn=128,
            dtype=jnp.bfloat16,
    ):
        keys = jax.random.split(key, 9)
        self.pre_conv = eqx.nn.Sequential(
            [
                SpectralConv2d(keys[0], 3, 2*chn, 3, padding=1, dtype=dtype),
                eqx.nn.Lambda(jax.nn.relu),
                SpectralConv2d(keys[1], 2*chn, 2*chn, 3, padding=1, dtype=dtype),
                eqx.nn.Lambda(eqx.nn.AvgPool2d(2))
            ]
        )
        self.pre_skip = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(eqx.nn.AvgPool2d(2)),
                SpectralConv2d(keys[2], 3, 2*chn, 1, padding=0, dtype=dtype)
            ]
        )
        self.conv_1 = DResBlock(keys[3], 2*chn, 4*chn, dtype=dtype)
        self.conv_2 = eqx.nn.Sequential(
            [
                DResBlock(keys[4], 4*chn, 8*chn, dtype=dtype),
                DResBlock(keys[5], 8*chn, 16*chn, dtype=dtype),
                DResBlock(keys[6], 16*chn, 16*chn, dtype=dtype)
            ]
        )
        self.linear = eqx.nn.SpectralNorm(
            eqx.nn.Linear(16*chn, 1, key=keys[7], dtype=dtype),
            weight_name="weight",
            key=keys[8]
        )

    def __call__(self, x, state):
        out = x
        out, state = eqx.filter_vmap(self.pre_conv, in_axes=(1, None), out_axes=(1, None))(out, state)
        skip = x
        skip, state = eqx.filter_vmap(self.pre_skip, in_axes=(1, None), out_axes=(1, None))(x, state)
        out = out + skip
        out, state = eqx.filter_vmap(self.conv_1, in_axes=(1, None), out_axes=(1, None))(out, state)
        out, state = eqx.filter_vmap(self.conv_2, in_axes=(1, None), out_axes=(1, None))(out, state)
        out = jax.nn.relu(out)
        out = jnp.sum(out, axis=(2, 3))
        out, state = eqx.filter_vmap(self.linear, in_axes=(1, None), out_axes=(1, None))(out, state)
        return jnp.mean(out), state

class TemporalDiscriminator(eqx.Module):
    pre_conv: eqx.nn.Sequential
    pre_skip: SpectralConv3d
    conv_1: D3DResBlock
    conv_2: eqx.nn.Sequential
    linear: eqx.nn.SpectralNorm[eqx.nn.Linear]

    def __init__(
            self, 
            key,
            chn=128,
            dtype=jnp.bfloat16,
    ):
        keys = jax.random.split(key, 9)
        self.pre_conv = eqx.nn.Sequential(
            [
                SpectralConv3d(keys[0], 3, 2*chn, 3, padding=1, dtype=dtype),
                eqx.nn.Lambda(jax.nn.relu),
                SpectralConv3d(keys[1], 2*chn, 2*chn, 3, padding=1, dtype=dtype),
                eqx.nn.Lambda(eqx.nn.AvgPool3d(2))
            ]
        )
        self.pre_skip = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(eqx.nn.AvgPool3d(2)),
                SpectralConv3d(keys[2], 3, 2*chn, 1, padding=0, dtype=dtype)
            ]
        )
        self.conv_1 = D3DResBlock(keys[3], 2*chn, 4*chn, dtype=dtype)
        self.conv_2 = eqx.nn.Sequential(
            [
                DResBlock(keys[4], 4*chn, 8*chn, dtype=dtype),
                DResBlock(keys[5], 8*chn, 16*chn, dtype=dtype),
                DResBlock(keys[6], 16*chn, 16*chn, dtype=dtype)
            ]
        )
        self.linear = eqx.nn.SpectralNorm(
            eqx.nn.Linear(16*chn, 1, key=keys[7], dtype=dtype),
            weight_name="weight",
            key=keys[8]
        )

    def __call__(self, x, state):
        out = x
        out, state = self.pre_conv(out, state)
        skip = x
        skip, state = self.pre_skip(x, state)
        out = out + skip
        out, state = self.conv_1(out, state)
        out, state = eqx.filter_vmap(self.conv_2, in_axes=(1, None), out_axes=(1, None))(out, state)
        out = jax.nn.relu(out)
        out = jnp.sum(out, axis=(2, 3))
        out, state = eqx.filter_vmap(self.linear, in_axes=(1, None), out_axes=(1, None))(out, state)
        return jnp.mean(out), state

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    subkey1, subkey2 = jax.random.split(subkey)
    x1 = jax.random.normal(subkey1, (2, 3, 8, 56, 56), dtype=jnp.bfloat16)
    x2 = jax.random.normal(subkey2, (2, 3, 8, 56, 56), dtype=jnp.bfloat16)
    model, state = eqx.nn.make_with_state(TemporalDiscriminator)(key)
    inference_model = eqx.nn.inference_mode(model)
    inference_model = eqx.Partial(inference_model, state=state)
    inference_model_jit = jax.jit(jax.vmap(inference_model, in_axes=0, out_axes=0))
    y, _ = inference_model_jit(x1)
    print(y)
    print(y.shape)
    y, _ = inference_model_jit(x2)
    print(y)
    print(y.shape)