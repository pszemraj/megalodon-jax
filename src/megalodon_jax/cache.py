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
"""Shared structural and timeline validation for streaming model caches."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool

from megalodon_jax.config import MegalodonConfig
from megalodon_jax.types import LayerCache, ModelCache, NormState

CACHE_INVARIANT_MESSAGE = (
    "cache must be either a sparse zero-history initializer or a complete, "
    "timeline-aligned continuation"
)


def _require_array(name: str, value: Array, shape: tuple[int, ...], dtype: jnp.dtype) -> None:
    """Validate static array schema available during eager execution and tracing.

    :param str name: Array name used in validation errors.
    :param Array value: Array to validate.
    :param tuple[int, ...] shape: Required array shape.
    :param jnp.dtype dtype: Required array dtype.
    """
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
    if jnp.dtype(value.dtype) != jnp.dtype(dtype):
        raise ValueError(f"{name} must have dtype {jnp.dtype(dtype)}, got {value.dtype}")


def _any(predicates: list[Array]) -> Bool[Array, ""]:
    """Combine scalar predicates without requiring a nonempty list.

    :param list[Array] predicates: Scalar predicates to combine.
    :return Bool[Array, ""]: Whether any predicate is true.
    """
    if not predicates:
        return jnp.asarray(False)
    return jnp.any(jnp.stack([jnp.asarray(predicate) for predicate in predicates]))


def layer_cache_invariant_violation(
    cache: LayerCache,
    *,
    batch_size: int,
    cache_capacity: int,
    num_heads: int,
    head_dim: int,
    value_head_dim: int,
    model_dim: int,
    cema_ndim: int,
    norm_num_groups: int,
    compute_dtype: jnp.dtype,
    increment: int = 0,
) -> Bool[Array, ""]:
    """Return whether one layer cache has an invalid continuation timeline.

    A direct layer call accepts either the sparse zero-history representation
    (all optional components absent and ``position == 0``) or a complete
    continuation with aligned attention and normalization counts. Compact
    normalization values are always checked because their cost is independent
    of attention-cache capacity.

    :param LayerCache cache: Layer cache to validate.
    :param int batch_size: Expected batch size.
    :param int cache_capacity: Required attention ring capacity.
    :param int num_heads: Required number of attention heads.
    :param int head_dim: Required query/key width per head.
    :param int value_head_dim: Required value width per head.
    :param int model_dim: Required CEMA model dimension.
    :param int cema_ndim: Required number of CEMA orders.
    :param int norm_num_groups: Required number of TimestepNorm groups.
    :param jnp.dtype compute_dtype: Required attention cache dtype.
    :param int increment: Timeline increment to validate against int32 overflow.
    :return Bool[Array, ""]: Scalar predicate that is true for an invalid cache.
    :raises ValueError: If the cache has partial state or invalid static schema.
    """
    if increment < 0 or increment > jnp.iinfo(jnp.int32).max:
        raise ValueError(f"cache increment must fit non-negative int32, got {increment}")

    _require_array("cache.position", cache.position, (), jnp.int32)
    present = (cache.attn is not None, cache.norm is not None, cache.ema is not None)
    if not any(present):
        return cache.position != 0
    if not all(present):
        raise ValueError(CACHE_INVARIANT_MESSAGE)

    assert cache.attn is not None
    assert cache.norm is not None
    assert cache.ema is not None
    _require_array(
        "cache.attn.k",
        cache.attn.k,
        (batch_size, cache_capacity, num_heads, head_dim),
        compute_dtype,
    )
    _require_array(
        "cache.attn.v",
        cache.attn.v,
        (batch_size, cache_capacity, num_heads, value_head_dim),
        compute_dtype,
    )
    _require_array("cache.attn.count", cache.attn.count, (), jnp.int32)
    _require_array("cache.norm.count", cache.norm.count, (batch_size,), jnp.int32)
    norm_shape = (batch_size, norm_num_groups)
    _require_array("cache.norm.mean", cache.norm.mean, norm_shape, jnp.float32)
    _require_array("cache.norm.var", cache.norm.var, norm_shape, jnp.float32)
    _require_array(
        "cache.ema.h",
        cache.ema.h,
        (batch_size, model_dim, cema_ndim),
        jnp.complex64,
    )

    return _any(
        [
            cache.position < 0,
            cache.attn.count != cache.position,
            jnp.any(cache.norm.count != cache.position),
            cache.position > jnp.iinfo(jnp.int32).max - increment,
            jnp.any(~jnp.isfinite(cache.norm.mean)),
            jnp.any(~jnp.isfinite(cache.norm.var)),
            jnp.any(cache.norm.var < 0.0),
            # Allocated history at position zero is not the sparse initializer.
            cache.position == 0,
        ]
    )


def classify_model_cache(
    cache: ModelCache,
    *,
    num_layers: int,
) -> Literal["sparse", "complete"]:
    """Classify one of the two executable top-level cache schemas.

    A sparse model cache uses ``None`` for every layer and for final norm. A
    complete continuation allocates every component in every layer plus final
    norm. Direct ``LayerCache()`` use remains a separate lower-level contract.

    :param ModelCache cache: Cache structure to classify.
    :param int num_layers: Required number of model layers.
    :raises ValueError: If the cache is partial or has the wrong layer count.
    :return Literal["sparse", "complete"]: Valid cache schema.
    """
    if len(cache.layer_caches) != num_layers:
        raise ValueError(
            f"cache has {len(cache.layer_caches)} layer entries, expected {num_layers}"
        )

    if cache.final_norm is None and all(layer is None for layer in cache.layer_caches):
        return "sparse"

    if cache.final_norm is not None and all(
        layer is not None
        and layer.attn is not None
        and layer.norm is not None
        and layer.ema is not None
        for layer in cache.layer_caches
    ):
        return "complete"

    raise ValueError(CACHE_INVARIANT_MESSAGE)


def cache_invariant_violation(
    cache: ModelCache,
    config: MegalodonConfig,
    *,
    batch_size: int | None = None,
    increment: int = 0,
    check_full_payload: bool = False,
) -> Bool[Array, ""]:
    """Return whether a cache violates the streaming-state coherence contract.

    Static schema errors raise immediately. Dynamic values produce one scalar
    predicate suitable for ``eqx.error_if`` under JIT or host evaluation during
    persistence. Compact normalization and EMA values are always checked.
    Attention K/V finiteness is optional because scanning the ring on every
    decode step would make model-entry validation proportional to cache
    capacity; persistence enables it before data crosses a trust boundary.

    :param ModelCache cache: Cache to validate.
    :param MegalodonConfig config: Model configuration defining the cache schema.
    :param int | None batch_size: Expected batch size, or None to infer it.
    :param int increment: Timeline increment to validate against int32 overflow.
    :param bool check_full_payload: Whether to validate attention K/V contents.
    :return Bool[Array, ""]: Scalar predicate that is true for an invalid cache.
    """
    if increment < 0 or increment > jnp.iinfo(jnp.int32).max:
        raise ValueError(f"cache increment must fit non-negative int32, got {increment}")
    cache_kind = classify_model_cache(cache, num_layers=config.num_layers)
    if cache_kind == "sparse":
        return jnp.asarray(False)

    bound_batch = batch_size

    def bind_batch(name: str, size: int) -> int:
        """Require one batch width across every allocated state component.

        :param str name: State component name used in validation errors.
        :param int size: Batch width of the state component.
        :return int: Validated batch width.
        """
        nonlocal bound_batch
        if bound_batch is None:
            bound_batch = size
        elif bound_batch != size:
            raise ValueError(f"cache batch mismatch at {name}: {size} != {bound_batch}")
        return size

    positions: list[Array] = []
    attention_counts: list[Array] = []
    attention_arrays: list[Array] = []
    norm_states: list[NormState] = []
    ema_states: list[Array] = []
    for index, layer in enumerate(cache.layer_caches):
        prefix = f"cache.layers.{index}"
        assert layer is not None
        _require_array(f"{prefix}.position", layer.position, (), jnp.int32)
        positions.append(layer.position)

        assert layer.attn is not None
        assert layer.norm is not None
        assert layer.ema is not None
        batch = bind_batch(f"{prefix}.attn.k", layer.attn.k.shape[0])
        _require_array(
            f"{prefix}.attn.k",
            layer.attn.k,
            (batch, config.cache_capacity, config.num_heads, config.head_dim),
            config.compute_dtype,
        )
        _require_array(
            f"{prefix}.attn.v",
            layer.attn.v,
            (batch, config.cache_capacity, config.num_heads, config.value_head_dim),
            config.compute_dtype,
        )
        _require_array(f"{prefix}.attn.count", layer.attn.count, (), jnp.int32)
        attention_counts.append(layer.attn.count)
        attention_arrays.extend((layer.attn.k, layer.attn.v))

        batch = bind_batch(f"{prefix}.norm.count", layer.norm.count.shape[0])
        _require_array(f"{prefix}.norm.count", layer.norm.count, (batch,), jnp.int32)
        _require_array(
            f"{prefix}.norm.mean",
            layer.norm.mean,
            (batch, config.norm_num_groups),
            jnp.float32,
        )
        _require_array(
            f"{prefix}.norm.var",
            layer.norm.var,
            (batch, config.norm_num_groups),
            jnp.float32,
        )
        norm_states.append(layer.norm)

        batch = bind_batch(f"{prefix}.ema.h", layer.ema.h.shape[0])
        _require_array(
            f"{prefix}.ema.h",
            layer.ema.h,
            (batch, config.model_dim, config.cema_ndim),
            jnp.complex64,
        )
        ema_states.append(layer.ema.h)

    assert cache.final_norm is not None
    batch = bind_batch("cache.final_norm.count", cache.final_norm.count.shape[0])
    _require_array("cache.final_norm.count", cache.final_norm.count, (batch,), jnp.int32)
    _require_array(
        "cache.final_norm.mean",
        cache.final_norm.mean,
        (batch, config.norm_num_groups),
        jnp.float32,
    )
    _require_array(
        "cache.final_norm.var",
        cache.final_norm.var,
        (batch, config.norm_num_groups),
        jnp.float32,
    )
    norm_states.append(cache.final_norm)

    counters = [*positions, *attention_counts, *(state.count for state in norm_states)]
    violations = [jnp.any(counter < 0) for counter in counters]
    for state in norm_states:
        violations.extend(
            (
                jnp.any(~jnp.isfinite(state.mean)),
                jnp.any(~jnp.isfinite(state.var)),
                jnp.any(state.var < 0.0),
            )
        )
    violations.extend(jnp.any(~jnp.isfinite(value)) for value in ema_states)
    if check_full_payload:
        violations.extend(jnp.any(~jnp.isfinite(value)) for value in attention_arrays)

    def pristine_violation() -> Bool[Array, ""]:
        """Check state that affects computation when the timeline is zero."""
        pristine_checks: list[Array] = []
        for state in norm_states:
            pristine_checks.extend(
                (
                    jnp.any(state.count != 0),
                    jnp.any(state.mean != 0.0),
                    jnp.any(state.var != 1.0),
                )
            )
        pristine_checks.extend(jnp.any(state != 0.0) for state in ema_states)
        return _any(pristine_checks)

    if positions:
        timeline = positions[0]
    elif bound_batch is not None and bound_batch > 0:
        timeline = cache.final_norm.count[0]
    else:
        timeline = jnp.asarray(0, dtype=jnp.int32)

    violations.extend(position != timeline for position in positions)
    violations.extend(count != timeline for count in attention_counts)
    violations.extend(jnp.any(state.count != timeline) for state in norm_states)
    violations.append(timeline > jnp.iinfo(jnp.int32).max - increment)
    if config.num_layers == 0:
        violations.append((timeline == 0) & pristine_violation())
    else:
        # Allocated per-layer histories are never the canonical empty cache.
        violations.append(timeline == 0)

    return _any(violations)


def validate_model_cache_host(cache: ModelCache, config: MegalodonConfig) -> None:
    """Raise ``ValueError`` when a concrete cache violates shared invariants.

    :param ModelCache cache: Concrete cache to validate on the host.
    :param MegalodonConfig config: Model configuration defining the cache schema.
    :raises ValueError: If the cache violates a structural or value invariant.
    """
    violation = cache_invariant_violation(cache, config, check_full_payload=True)
    if bool(np.asarray(jax.device_get(violation))):
        raise ValueError(CACHE_INVARIANT_MESSAGE)
