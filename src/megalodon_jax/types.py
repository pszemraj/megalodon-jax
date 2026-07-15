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
"""Type definitions for Megalodon cache and state.

CRITICAL: All position/count fields are JAX arrays (not Python ints).
Python ints inside jax.lax.scan become static and cause recompilation on each new value.

All cache/state dataclasses are registered as JAX pytrees to work with jit/scan.
"""

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Complex, Float, Int


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("k", "v", "count"),
    meta_fields=(),
)
@dataclass
class AttentionCache:
    """Cache for streaming attention.

    Stores key/value tensors in fixed-capacity buffers.
    Use `count` for the total tokens processed (absolute position after last token).
    Buffer capacity is `k.shape[1]`; validity is inferred from `count` and ring
    indexing rather than an explicit mask. The buffer is treated as a circular
    ring indexed by absolute position modulo the cache capacity.

    Note: Unlike the PyTorch reference, this cache does not expose length/start_index
    properties because they would be misleading. In JAX, caches use fixed-size buffers
    with internal validity masking, so buffer capacity != valid cached length.
    """

    k: Float[Array, "batch seq heads head_dim"]
    v: Float[Array, "batch seq heads value_head_dim"]
    count: Int[Array, ""]  # JAX scalar - total tokens seen


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("count", "mean", "var"),
    meta_fields=(),
)
@dataclass
class NormState:
    """Running statistics for TimestepNorm (Welford's algorithm).

    Tracks cumulative count, mean, and variance per group for streaming
    normalization that only uses past context. Mean/variance are stored in
    float32 for numerical stability, regardless of activation dtype.
    """

    count: Int[Array, "batch"]  # tokens seen per batch element
    mean: Float[Array, "batch groups"]
    var: Float[Array, "batch groups"]  # running variance estimate


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("h",),
    meta_fields=(),
)
@dataclass
class EMAState:
    """Complex exponential moving average hidden state.

    The EMA module maintains a complex hidden state that accumulates
    information across the sequence via recurrence: h[t] = q*h[t-1] + p*x[t]
    """

    h: Complex[Array, "batch dim ndim"]


def _default_position() -> Int[Array, ""]:
    """Create default position counter as JAX scalar.

    :return Int[Array, ""]: JAX scalar position counter initialized to zero.
    """
    return jnp.array(0, dtype=jnp.int32)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("attn", "norm", "ema", "position"),
    meta_fields=(),
)
@dataclass
class LayerCache:
    """Combined cache for a single transformer layer.

    Groups attention cache, norm state, and EMA state. ``position`` is retained
    as a schema-level compatibility mirror of ``AttentionCache.count``; cached
    attention uses ``attn.count`` as the authoritative RoPE/ring position.
    """

    attn: AttentionCache | None = None
    norm: NormState | None = None
    ema: EMAState | None = None
    position: Int[Array, ""] = field(default_factory=_default_position)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("layer_caches", "final_norm"),
    meta_fields=(),
)
@dataclass
class ModelCache:
    """Full model cache: layer caches + final norm state.

    This cache structure holds all streaming state for the model:
    - One LayerCache per decoder layer (attention, norm, EMA state)
    - One NormState for the final TimestepNorm

    Note: layer_caches must be a tuple (not list) for JAX pytree compatibility.
    """

    layer_caches: tuple[LayerCache | None, ...]
    final_norm: NormState | None = None


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("cache", "next_logits", "finished"),
    meta_fields=("eos_token_id",),
)
@dataclass
class GenerationState:
    """Complete state required to resume autoregressive generation.

    ``cache`` includes every token emitted so far, while ``next_logits`` are
    the logits produced by the final cached token and are therefore ready for
    the next sampling decision. ``finished`` preserves per-row EOS status for
    fixed-shape batched continuation. The EOS token is static metadata so a
    resumed call cannot silently change the meaning of ``finished``.
    """

    cache: ModelCache
    next_logits: Float[Array, "batch vocab"]
    finished: Bool[Array, "batch"]
    eos_token_id: int | None = None
