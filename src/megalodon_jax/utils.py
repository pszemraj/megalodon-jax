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
"""Utility functions for Megalodon JAX."""

from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from megalodon_jax.config import InitMode

T = TypeVar("T", bound=eqx.Module)


def _initializer_sample_dtype(dtype: jnp.dtype) -> jnp.dtype:
    """Use FP32 random draws when the persistent destination is BF16.

    :param jnp.dtype dtype: Requested persistent parameter dtype.
    :return jnp.dtype: Random sampling dtype.
    """
    return jnp.float32 if jnp.dtype(dtype) == jnp.dtype(jnp.bfloat16) else dtype


def _is_linear(x: Any) -> bool:
    """Check if x is an Equinox Linear layer.

    Used as a leaf predicate for JAX tree traversal to identify
    layers that should have their weights reinitialized.

    :param Any x: Pytree node to check.
    :return bool: True if x is an eqx.nn.Linear instance.
    """
    return isinstance(x, eqx.nn.Linear)


def get_initializer(
    mode: InitMode, dim: int | None = None
) -> Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], Array]:
    """Return the exact released initializer for an internal weight tensor.

    :param InitMode mode: Initialization mode.
    :param int | None dim: Dimension for gaussian stddev; None uses stddev=1.0.
    :return Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], Array]: Initializer.

    Equinox Linear weights are stored as (out_features, in_features), so fan-in
    is shape[-1] and fan-out is shape[-2]. The released ``he`` mode is
    ``kaiming_normal_(a=sqrt(5))``, not canonical ReLU He normal.
    """

    def require_matrix(shape: tuple[int, ...]) -> tuple[int, int]:
        """Return fan-in and fan-out for a matrix-shaped parameter.

        :param tuple[int, ...] shape: Parameter shape ending in output and input axes.
        :raises ValueError: If ``shape`` does not describe a matrix.
        :return tuple[int, int]: Fan-in and fan-out.
        """
        if len(shape) < 2:
            raise ValueError(f"{mode} initializer requires a matrix shape, got {shape}")
        return int(shape[-1]), int(shape[-2])

    if mode == "he":

        def he(key: PRNGKeyArray, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32) -> Array:
            """Draw source-compatible Kaiming-normal values.

            :param PRNGKeyArray key: Random key.
            :param tuple[int, ...] shape: Output array shape.
            :param jnp.dtype dtype: Output dtype.
            :return Array: Initialized array.
            """
            fan_in, _ = require_matrix(shape)
            std = 1.0 / jnp.sqrt(3.0 * fan_in)
            values = jax.random.normal(key, shape, dtype=_initializer_sample_dtype(dtype))
            return (values * std).astype(dtype)

        return he

    if mode == "xavier":

        def xavier(
            key: PRNGKeyArray, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
        ) -> Array:
            """Draw Xavier-uniform values.

            :param PRNGKeyArray key: Random key.
            :param tuple[int, ...] shape: Output array shape.
            :param jnp.dtype dtype: Output dtype.
            :return Array: Initialized array.
            """
            fan_in, fan_out = require_matrix(shape)
            bound = jnp.sqrt(6.0 / (fan_in + fan_out))
            values = jax.random.uniform(
                key,
                shape,
                dtype=_initializer_sample_dtype(dtype),
                minval=-bound,
                maxval=bound,
            )
            return values.astype(dtype)

        return xavier

    if mode == "bert":

        def bert(
            key: PRNGKeyArray, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
        ) -> Array:
            """Draw normal values with the BERT standard deviation.

            :param PRNGKeyArray key: Random key.
            :param tuple[int, ...] shape: Output array shape.
            :param jnp.dtype dtype: Output dtype.
            :return Array: Initialized array.
            """
            values = jax.random.normal(key, shape, dtype=_initializer_sample_dtype(dtype))
            return (values * 0.02).astype(dtype)

        return bert

    if mode == "gaussian":
        std = 1.0 if dim is None else 1.0 / jnp.sqrt(dim)

        def gaussian(
            key: PRNGKeyArray, shape: tuple[int, ...], dtype: jnp.dtype = jnp.float32
        ) -> Array:
            """Draw truncated-normal values using the configured deviation.

            :param PRNGKeyArray key: Random key.
            :param tuple[int, ...] shape: Output array shape.
            :param jnp.dtype dtype: Output dtype.
            :return Array: Initialized array.
            """
            values = jax.random.truncated_normal(
                key,
                lower=-3.0,
                upper=3.0,
                shape=shape,
                dtype=_initializer_sample_dtype(dtype),
            )
            return (values * jnp.asarray(std, dtype=values.dtype)).astype(dtype)

        return gaussian

    raise ValueError(f"Unknown init mode: {mode}")


def get_boundary_initializer(
    model_dim: int,
) -> Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], Array]:
    """Return the fixed embedding/output-head initializer from upstream.

    :param int model_dim: Model width controlling the initializer standard deviation.
    :return Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], Array]: Boundary initializer.
    """
    if model_dim <= 0:
        raise ValueError(f"model_dim must be positive, got {model_dim}")
    return get_initializer("gaussian", dim=model_dim)


def reinit_linear_weights(
    model: T,
    mode: InitMode,
    key: PRNGKeyArray,
    dim: int | None = None,
) -> T:
    """Reinitialize all Linear layer weights in a model using the specified init mode.

    This function traverses the Equinox module tree and replaces all Linear layer
    weights with freshly initialized values. Biases are reset to zeros.

    :param T model: Equinox module to reinitialize.
    :param InitMode mode: Initialization mode for weights.
    :param PRNGKeyArray key: PRNG key for random initialization.
    :param int | None dim: Dimension for gaussian stddev; None uses stddev=1.0.
    :return T: Model with reinitialized Linear weights.
    """
    init_fn = get_initializer(mode, dim)

    # Find all Linear layers using module-level predicate
    leaves, treedef = jax.tree_util.tree_flatten(model, is_leaf=_is_linear)

    # Count linear layers for key splitting
    num_linears = sum(1 for leaf in leaves if _is_linear(leaf))
    if num_linears == 0:
        return model

    keys = jax.random.split(key, num_linears)
    key_idx = 0

    new_leaves = []
    for leaf in leaves:
        if _is_linear(leaf):
            # Get weight shape and dtype
            weight = leaf.weight
            shape = weight.shape
            dtype = weight.dtype

            # For gaussian mode, use the provided dim (or None for std=1.0, matching PyTorch)
            # For other modes, use the shared init_fn
            if mode == "gaussian":
                local_init = get_initializer(mode, dim)
            else:
                local_init = init_fn

            # Initialize new weight
            new_weight = local_init(keys[key_idx], shape, dtype)
            key_idx += 1

            # Reset bias to zeros if present
            if leaf.bias is not None:
                new_bias = jnp.zeros_like(leaf.bias)
                new_leaf = eqx.tree_at(
                    lambda linear: (linear.weight, linear.bias), leaf, (new_weight, new_bias)
                )
            else:
                new_leaf = eqx.tree_at(lambda linear: linear.weight, leaf, new_weight)

            new_leaves.append(new_leaf)
        else:
            new_leaves.append(leaf)

    return jax.tree_util.tree_unflatten(treedef, new_leaves)
