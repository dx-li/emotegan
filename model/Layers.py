import equinox as eqx
import jax
import jax.numpy as jnp
import optax  # https://github.com/deepmind/optax
from jaxtyping import Array, Float, Int, PyTree  # https://github.com/google/jaxtyping

class SpectralConv2d(eqx.nn.StatefulLayer):
    spec_conv: eqx.nn.SpectralNorm[eqx.nn.Conv2d]

    def __init__(
            self,
            key,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            dtype=jnp.bfloat16
    ):
        key1, key2 = jax.random.split(key)
        self.spec_conv = eqx.nn.SpectralNorm(
            eqx.nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=stride, 
                padding=padding, 
                key=key1,
                dtype=dtype
            ),
            weight_name="weight",
            key=key2
        )

    def __call__(self, x, state, key=None, inference=None):
        return self.spec_conv(x, state)