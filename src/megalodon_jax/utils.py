"""Utility functions for Megalodon JAX."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from megalodon_jax.config import InitMode


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
            - "gaussian": Truncated normal with stddev=1/sqrt(dim)
        dim: Dimension for computing stddev in "gaussian" mode.

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
        if dim is None:
            raise ValueError("dim must be provided for 'gaussian' initialization")
        std = 1.0 / jnp.sqrt(dim)
        return jax.nn.initializers.truncated_normal(stddev=std, lower=-3 * std, upper=3 * std)

    raise ValueError(f"Unknown init mode: {mode}")
