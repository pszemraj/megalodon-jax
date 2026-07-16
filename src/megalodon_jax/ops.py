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
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


def inverted_dropout(x: Array, rate: float, key: PRNGKeyArray) -> Array:
    """Apply inverted dropout while preserving the input dtype.

    :param Array x: Activations to mask.
    :param float rate: Probability of dropping each activation.
    :param PRNGKeyArray key: Random key used to sample the keep mask.
    :return Array: Masked activations rescaled by the inverse keep probability.
    """
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    inv_keep = jnp.asarray(1.0 / (1.0 - rate), dtype=x.dtype)
    return jnp.where(keep, x * inv_keep, jnp.zeros((), dtype=x.dtype))


def dot_precision(compute_dtype: jnp.dtype) -> jax.lax.Precision | None:
    """Select full FP32 products without constraining native BF16 GEMMs.

    :param jnp.dtype compute_dtype: Matrix-product compute dtype.
    :return jax.lax.Precision | None: Highest precision for FP32, otherwise the backend default.
    """
    if jnp.dtype(compute_dtype) == jnp.dtype(jnp.float32):
        return jax.lax.Precision.HIGHEST
    return None


def bf16_f32_dot_precision(
    compute_dtype: jnp.dtype,
) -> jax.lax.Precision | jax.lax.DotAlgorithmPreset | None:
    """Select explicit FP32 accumulation for BF16 result-buffer contractions.

    :param jnp.dtype compute_dtype: Matrix-product compute dtype.
    :return jax.lax.Precision | jax.lax.DotAlgorithmPreset | None: BF16 operands with
        FP32 accumulation for BF16 compute, otherwise the existing precision policy.
    """
    if jnp.dtype(compute_dtype) == jnp.dtype(jnp.bfloat16):
        return jax.lax.DotAlgorithmPreset.BF16_BF16_F32
    return dot_precision(compute_dtype)


def _matmul_3d(
    x: Array,
    weight: Array,
    compute_dtype: jnp.dtype,
    accum_dtype: jnp.dtype,
    output_dtype: jnp.dtype,
    bias: Array | None = None,
) -> Array:
    """Apply the shared mixed-precision matrix product used by 3D projections.

    :param Array x: Input tensor with its feature dimension last.
    :param Array weight: Output-by-input weight matrix.
    :param jnp.dtype compute_dtype: Operand dtype used by the matrix product.
    :param jnp.dtype accum_dtype: Preferred accumulation dtype.
    :param jnp.dtype output_dtype: Dtype of the returned tensor.
    :param Array | None bias: Optional output-feature bias.
    :return Array: Projected tensor in ``output_dtype``.
    """
    direct_bf16_result = (
        bias is None
        and jnp.dtype(compute_dtype) == jnp.dtype(jnp.bfloat16)
        and jnp.dtype(output_dtype) == jnp.dtype(jnp.bfloat16)
    )
    y = jnp.matmul(
        x.astype(compute_dtype),
        weight.astype(compute_dtype).T,
        precision=(
            bf16_f32_dot_precision(compute_dtype)
            if direct_bf16_result
            else dot_precision(compute_dtype)
        ),
        preferred_element_type=None if direct_bf16_result else accum_dtype,
    )
    if bias is not None:
        # Match upstream Linear semantics by adding bias before the final downcast.
        y = y + bias.astype(compute_dtype)
    return y.astype(output_dtype)


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
    return _matmul_3d(
        x,
        linear.weight,
        compute_dtype,
        accum_dtype,
        compute_dtype,
        linear.bias,
    )


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
    return _matmul_3d(
        x,
        weight,
        compute_dtype,
        accum_dtype,
        compute_dtype if output_dtype is None else output_dtype,
    )
