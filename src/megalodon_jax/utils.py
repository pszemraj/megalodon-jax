"""Utility functions for Megalodon JAX."""

from collections.abc import Callable
from typing import Any, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from megalodon_jax.config import InitMode

T = TypeVar("T", bound=eqx.Module)


def _is_linear(x: Any) -> bool:
    """Check if x is an Equinox Linear layer.

    Used as a leaf predicate for JAX tree traversal to identify
    layers that should have their weights reinitialized.

    Args:
        x: Any pytree node to check.

    Returns:
        True if x is an eqx.nn.Linear instance, False otherwise.
    """
    return isinstance(x, eqx.nn.Linear)


def get_initializer(
    mode: InitMode, dim: int | None = None
) -> Callable[[PRNGKeyArray, tuple[int, ...], jnp.dtype], Array]:
    """Get a JAX-compatible weight initializer.

    Args:
        mode: Initialization mode. One of:
            - "none": Zero initialization
            - "he": He normal (variance scaling for ReLU)
            - "xavier": Xavier uniform (Glorot)
            - "bert": Normal with stddev=0.02
            - "gaussian": Truncated normal with stddev=1/sqrt(dim) when dim is set,
              or stddev=1.0 when dim is None (PyTorch parity).
        dim: Dimension for computing stddev in "gaussian" mode; if None uses stddev=1.0.

    Returns:
        Callable that takes (key, shape, dtype) and returns an array.
    """
    if mode == "none":
        return lambda key, shape, dtype: jnp.zeros(shape, dtype=dtype)

    if mode == "he":
        return jax.nn.initializers.he_normal()

    if mode == "xavier":
        return jax.nn.initializers.glorot_uniform()

    if mode == "bert":
        return jax.nn.initializers.normal(stddev=0.02)

    if mode == "gaussian":
        # Match PyTorch: std=1.0 when dim is None, else 1/sqrt(dim)
        if dim is None:
            std = 1.0
        else:
            std = 1.0 / jnp.sqrt(dim)
        # Bounds are in standard-normal units (before scaling by stddev).
        # Final output range: lower*stddev < x < upper*stddev = (-3*std, 3*std)
        return jax.nn.initializers.truncated_normal(stddev=std, lower=-3.0, upper=3.0)

    raise ValueError(f"Unknown init mode: {mode}")


def reinit_linear_weights(
    model: T,
    mode: InitMode,
    key: PRNGKeyArray,
    dim: int | None = None,
) -> T:
    """Reinitialize all Linear layer weights in a model using the specified init mode.

    This function traverses the Equinox module tree and replaces all Linear layer
    weights with freshly initialized values. Biases are reset to zeros.

    Args:
        model: Equinox module to reinitialize.
        mode: Initialization mode for weights.
        key: PRNG key for random initialization.
        dim: Dimension for "gaussian" mode (optional; None uses stddev=1.0).

    Returns:
        New model with reinitialized Linear weights.

    Example:
        >>> key = jax.random.PRNGKey(42)
        >>> model = MegalodonAttention(...)
        >>> model = reinit_linear_weights(model, "he", key)
    """
    if mode == "none":
        return model  # Skip reinitialization

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
