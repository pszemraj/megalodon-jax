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
"""Low-level math ops with precision policy support."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def linear_3d(
    linear: eqx.nn.Linear,
    x: Float[Array, "batch seq in_dim"],
    compute_dtype: jnp.dtype,
    accum_dtype: jnp.dtype,
) -> Float[Array, "batch seq out_dim"]:
    """Apply an Equinox Linear over a (batch, seq, dim) tensor.

    :param eqx.nn.Linear linear: Linear module to apply.
    :param jax.Array x: Input tensor of shape (batch, seq, in_dim).
    :param jnp.dtype compute_dtype: Compute dtype for matmul and output.
    :param jnp.dtype accum_dtype: Accumulation dtype for GEMM.
    :return jax.Array: Output tensor of shape (batch, seq, out_dim).
    """
    x_c = x.astype(compute_dtype)
    w_c = linear.weight.astype(compute_dtype)
    y = jnp.matmul(x_c, w_c.T, preferred_element_type=accum_dtype)
    if linear.bias is not None:
        y = y + linear.bias.astype(compute_dtype)
    # Downcast from accum_dtype (e.g., fp32) to compute_dtype (e.g., bf16).
    return y.astype(compute_dtype)


def matmul_3d_weight(
    x: Float[Array, "batch seq in_dim"],
    weight: Float[Array, "out_dim in_dim"],
    compute_dtype: jnp.dtype,
    accum_dtype: jnp.dtype,
    output_dtype: jnp.dtype | None = None,
) -> Float[Array, "batch seq out_dim"]:
    """Apply a weight matrix to a (batch, seq, dim) tensor with compute policy.

    :param Float[Array, "batch seq in_dim"] x: Input tensor.
    :param Float[Array, "out_dim in_dim"] weight: Weight matrix (out_dim, in_dim).
    :param jnp.dtype compute_dtype: Compute dtype for matmul and output.
    :param jnp.dtype accum_dtype: Accumulation dtype for GEMM.
    :param jnp.dtype | None output_dtype: Output dtype; defaults to compute_dtype.
    :return Float[Array, "batch seq out_dim"]: Output tensor.
    """
    x_c = x.astype(compute_dtype)
    w_c = weight.astype(compute_dtype)
    y = jnp.matmul(x_c, w_c.T, preferred_element_type=accum_dtype)
    return y.astype(compute_dtype if output_dtype is None else output_dtype)
