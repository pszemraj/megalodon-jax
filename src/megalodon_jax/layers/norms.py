# Copyright 2025 Peter Szemraj.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Normalization layers for Megalodon JAX."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value and optionally applies a learnable scale.
    The gamma parameter uses a (gamma + 1) parameterization so that
    initializing gamma to zeros gives an effective scale of 1.

    Attributes:
        dim: Feature dimension.
        eps: Small constant for numerical stability.
        affine: Whether to include learnable scale parameter.
        gamma: Learnable scale parameter (shape: dim), or None if affine=False.
    """

    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)
    gamma: Float[Array, "dim"] | None

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        affine: bool = True,
        *,
        key: PRNGKeyArray | None = None,
    ):
        """Initialize RMSNorm.

        :param int dim: Feature dimension.
        :param float eps: Numerical stability epsilon.
        :param bool affine: Whether to include learnable scale.
        :param PRNGKeyArray | None key: PRNG key (unused).
        :return None: None.
        """
        del key  # unused
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = jnp.zeros(dim)
        else:
            self.gamma = None

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Apply RMS normalization.

        :param Float[Array, "... dim"] x: Input tensor with feature dimension last.
        :return Float[Array, "... dim"]: Normalized tensor.
        """
        # Compute RMS in fp32 to avoid bf16 overflow on x**2
        # (bf16 max ~65504, so values > ~256 would overflow when squared)
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        x_normed = (x_f32 / rms).astype(x.dtype)
        # Apply scale if affine (cast gamma to input dtype to preserve bf16)
        if self.affine and self.gamma is not None:
            scale = (self.gamma + 1.0).astype(x.dtype)
            return x_normed * scale
        return x_normed


class BatchedLayerNorm(eqx.Module):
    """LayerNorm for batched inputs with normalization over the last dimension.

    This implementation accepts arbitrary leading dimensions and normalizes over
    the final axis, matching PyTorch LayerNorm semantics for [*, D] inputs.
    """

    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)
    weight: Float[Array, "dim"] | None
    bias: Float[Array, "dim"] | None

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        key: PRNGKeyArray | None = None,
    ):
        """Initialize BatchedLayerNorm.

        :param int dim: Feature dimension for normalization.
        :param float eps: Numerical stability epsilon.
        :param bool affine: Whether to include learnable scale and bias.
        :param PRNGKeyArray | None key: PRNG key (unused).
        :return None: None.
        """
        del key  # unused
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = jnp.ones((dim,), dtype=jnp.float32)
            self.bias = jnp.zeros((dim,), dtype=jnp.float32)
        else:
            self.weight = None
            self.bias = None

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Apply LayerNorm over the last dimension.

        :param Float[Array, "... dim"] x: Input tensor with feature dimension last.
        :return Float[Array, "... dim"]: Normalized tensor.
        """
        x_f32 = x.astype(jnp.float32)
        mean = jnp.mean(x_f32, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
        y = (x_f32 - mean) * jax.lax.rsqrt(var + self.eps)
        y = y.astype(x.dtype)
        if self.affine and self.weight is not None and self.bias is not None:
            weight = self.weight.astype(x.dtype)
            bias = self.bias.astype(x.dtype)
            y = y * weight + bias
        return y
