"""Independent cache-partition references used by regression and audit gates."""

from __future__ import annotations

from typing import Any

import numpy as np


def _max_abs_error(actual: Any, expected: Any) -> float:
    """Return the maximum absolute difference between array-like values.

    :param Any actual: Observed values.
    :param Any expected: Reference values.
    :return float: Maximum absolute elementwise error.
    """
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def cache_partition_errors(
    attention_window: int | None,
    partition: tuple[int, ...] = (1,) * 12,
    *,
    seed: int = 17,
) -> dict[str, float]:
    """Compare full and partitioned attention outputs and cache state.

    :param int | None attention_window: Sliding-window width, or ``None`` for
        faithful chunk-local attention.
    :param tuple[int, ...] partition: Positive call widths summing to 12.
    :param int seed: Deterministic JAX seed.
    :return dict[str, float]: Maximum errors for outputs and cache fields.
    """
    import jax
    import jax.numpy as jnp

    from megalodon_jax.layers.attention import ChunkedAttention

    length = 12
    if sum(partition) != length or any(width <= 0 for width in partition):
        raise ValueError(f"partition must contain positive widths summing to {length}")
    key = jax.random.PRNGKey(seed)
    k_module, kq, kk, kv = jax.random.split(key, 4)
    module = ChunkedAttention(
        num_heads=1,
        head_dim=4,
        value_head_dim=3,
        chunk_size=4,
        attention_window=attention_window,
        key=k_module,
    )
    q = jax.random.normal(kq, (1, length, 1, 4))
    k = jax.random.normal(kk, (1, length, 1, 4))
    v = jax.random.normal(kv, (1, length, 1, 3))
    full, full_cache, _ = module(q, k, v, return_cache=True)
    noncached, _, _ = module(q, k, v)
    if full_cache is None:
        raise AssertionError("full attention call did not return a cache")

    pieces = []
    cache = None
    start = 0
    for width in partition:
        stop = start + width
        part, cache, _ = module(
            q[:, start:stop],
            k[:, start:stop],
            v[:, start:stop],
            cache=cache,
            return_cache=True,
        )
        pieces.append(part)
        start = stop
    if cache is None:
        raise AssertionError("partitioned attention call did not return a cache")
    partitioned = jnp.concatenate(pieces, axis=1)
    return {
        "output": _max_abs_error(partitioned, full),
        "noncached_output": _max_abs_error(noncached, full),
        "cache_k": _max_abs_error(cache.k, full_cache.k),
        "cache_v": _max_abs_error(cache.v, full_cache.v),
        "cache_count": _max_abs_error(cache.count, full_cache.count),
    }
