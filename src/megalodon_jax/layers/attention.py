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
"""Attention primitives and modules for Megalodon JAX.

This module implements:
1. attention_single_chunk - pure causal attention with scale=1.0 (no temperature)
2. attention_multi_chunk - block-diagonal chunked attention for training
3. ChunkedAttention - module with RoPE and KV cache for streaming
4. MegalodonAttention - complete attention block with CEMA and gating
5. NormalizedFFN - pre-norm FFN with optional SwiGLU

Key design decisions:
- NO temperature scaling (scale=1.0) - normalized attention uses per-head RMS norm + affine
- Tensor layout: (batch, seq, heads, dim) matching PyTorch reference
- Cache stores already-rotated K for parity with PyTorch
- All counters are JAX arrays (not Python ints) to prevent recompilation
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.cache import CACHE_INVARIANT_MESSAGE, layer_cache_invariant_violation
from megalodon_jax.config import AttentionDropoutMode
from megalodon_jax.layers.complex_ema import ComplexEMA
from megalodon_jax.layers.norms import BatchedLayerNorm, RMSNorm, _rms_normalize
from megalodon_jax.layers.rotary import RotaryEmbedding, RotaryTable
from megalodon_jax.layers.segments import (
    SegmentMetadata,
    derive_segment_metadata,
)
from megalodon_jax.layers.timestep_norm import TimestepNorm
from megalodon_jax.ops import (
    bf16_f32_dot_precision,
    dot_precision,
    inverted_dropout,
    linear_3d,
)
from megalodon_jax.types import AttentionCache, EMAState, LayerCache

# -----------------------------------------------------------------------------
# Attention Primitives (Pure Functions)
# -----------------------------------------------------------------------------


def _validate_dropout_rate(name: str, value: float) -> None:
    """Reject probabilities that make inverted dropout undefined.

    :param str name: Configuration field name used in validation errors.
    :param float value: Dropout probability to validate.
    :raises ValueError: If value is outside the half-open interval ``[0, 1)``.
    """
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def _validate_dropout_mode(mode: AttentionDropoutMode) -> None:
    """Validate the supported attention dropout placement."""
    if mode not in ("post_softmax", "dropkey"):
        raise ValueError(
            f"attention dropout mode must be 'post_softmax' or 'dropkey', got {mode!r}"
        )


def attention_single_chunk(
    q: Float[Array, "batch seq heads head_dim"],
    k: Float[Array, "batch kv_seq heads head_dim"],
    v: Float[Array, "batch kv_seq heads value_dim"],
    kv_mask: Bool[Array, "batch kv_seq"] | None = None,
    qk_mask: Bool[Array, "batch seq kv_seq"] | None = None,
    accum_dtype: jnp.dtype = jnp.float32,
    softmax_dtype: jnp.dtype = jnp.float32,
    causal: bool = True,
    dropout_rate: float = 0.0,
    dropout_mode: AttentionDropoutMode = "post_softmax",
    deterministic: bool = True,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "batch seq heads value_dim"]:
    """Causal attention over a single chunk with normalized scaling.

    This implements normalized attention (scale=1.0), not standard scaled
    dot-product attention. The PyTorch reference explicitly sets scale=1.0 in
    ``F.scaled_dot_product_attention``.

    :param jax.Array q: Query tensor of shape (batch, seq, heads, head_dim).
    :param jax.Array k: Key tensor of shape (batch, kv_seq, heads, head_dim).
    :param jax.Array v: Value tensor of shape (batch, kv_seq, heads, value_dim).
    :param jax.Array | None kv_mask: Optional mask where True marks valid positions.
    :param jax.Array | None qk_mask: Optional per-query/per-key boolean mask.
    :param jnp.dtype accum_dtype: Accumulation dtype for attention matmuls.
    :param jnp.dtype softmax_dtype: Dtype used to evaluate attention softmax.
    :param bool causal: Whether to apply causal masking.
    :param float dropout_rate: Attention dropout rate.
    :param AttentionDropoutMode dropout_mode: Post-softmax dropout or DropKey.
    :param bool deterministic: If True, skip dropout.
    :param PRNGKeyArray | None key: PRNG key for dropout when deterministic is False.
    :raises ValueError: If dropout is enabled without a PRNG key.
    :return jax.Array: Output tensor of shape (batch, seq, heads, value_dim).
    """
    B, L_q, H, Dh = q.shape
    L_kv = k.shape[1]
    Dv = v.shape[-1]

    # Handle empty sequence
    if L_q == 0 or L_kv == 0:
        return jnp.zeros((B, L_q, H, Dv), dtype=q.dtype)
    _validate_dropout_rate("dropout_rate", dropout_rate)
    _validate_dropout_mode(dropout_mode)
    if not deterministic and dropout_rate > 0.0 and key is None:
        raise ValueError("PRNG key required for attention dropout")

    # Compute attention scores: (B, H, L_q, L_kv)
    # NO scaling by 1/sqrt(d_k) - this is normalized attention
    # Keep inputs in native dtype but accumulate in accum_dtype for stability/perf.
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q,
        k,
        precision=dot_precision(q.dtype),
        preferred_element_type=accum_dtype,
    )

    # Build attention mask
    # Use -inf so softmax(all masked) = NaN, triggering the NaN guard below
    neg_inf = -jnp.inf
    valid_edges = None
    if causal:
        # Align the last query with the last key. For cached attention this lets
        # each query see all prior keys; negative offsets correctly represent
        # queries that precede every available key when L_q > L_kv.
        q_pos = jnp.arange(L_q)[:, None]  # (L_q, 1)
        k_pos = jnp.arange(L_kv)[None, :]  # (1, L_kv)
        offset = L_kv - L_q
        causal_mask = k_pos <= (q_pos + offset)  # (L_q, L_kv)
        scores = jnp.where(causal_mask, scores, neg_inf)
        if L_q > L_kv:
            valid_edges = causal_mask[None, None, :, :]

    # Apply key/value padding mask if provided
    if kv_mask is not None:
        # kv_mask: (B, L_kv) with True for valid positions
        kv_mask_expanded = kv_mask[:, None, None, :]  # (B, 1, 1, L_kv)
        scores = jnp.where(kv_mask_expanded, scores, neg_inf)
        if valid_edges is None:
            valid_edges = (
                causal_mask[None, None, :, :]
                if causal
                else jnp.ones((1, 1, L_q, L_kv), dtype=jnp.bool_)
            )
        valid_edges = valid_edges & kv_mask_expanded
    if qk_mask is not None:
        qk_mask_expanded = qk_mask[:, None, :, :]
        scores = jnp.where(qk_mask_expanded, scores, neg_inf)
        if valid_edges is None:
            valid_edges = (
                causal_mask[None, None, :, :]
                if causal
                else jnp.ones((1, 1, L_q, L_kv), dtype=jnp.bool_)
            )
        valid_edges = valid_edges & qk_mask_expanded

    scores = scores.astype(softmax_dtype)
    if not deterministic and dropout_rate > 0.0 and dropout_mode == "dropkey":
        keep_mask = jax.random.bernoulli(key, 1.0 - dropout_rate, scores.shape)
        scores = jnp.where(keep_mask, scores, jnp.asarray(-jnp.inf, dtype=scores.dtype))
        if valid_edges is None:
            valid_edges = (
                causal_mask[None, None, :, :]
                if causal
                else jnp.ones((1, 1, L_q, L_kv), dtype=jnp.bool_)
            )
        valid_edges = valid_edges & keep_mask

    # Only structurally empty rows are zeroed. NaNs from valid scores remain
    # observable rather than being silently converted into plausible output.
    attn_weights = jax.nn.softmax(scores, axis=-1)
    if valid_edges is not None:
        has_valid_edge = jnp.any(valid_edges, axis=-1, keepdims=True)
        attn_weights = jnp.where(has_valid_edge, attn_weights, 0.0)

    # Match the released implementation's mixed-precision boundary: evaluate
    # softmax in its configured dtype, then return probabilities to the Q/K/V
    # compute dtype before dropout and the value contraction.
    attn_weights = attn_weights.astype(q.dtype)

    # Dropout if not deterministic
    if not deterministic and dropout_rate > 0.0 and dropout_mode == "post_softmax":
        attn_weights = inverted_dropout(attn_weights, dropout_rate, key)

    # Apply to values: (B, H, L_q, Dv) -> (B, L_q, H, Dv)
    out = jnp.einsum(
        "bhqk,bkhd->bqhd",
        attn_weights,
        v,
        precision=bf16_f32_dot_precision(q.dtype),
        preferred_element_type=(
            None if jnp.dtype(q.dtype) == jnp.dtype(jnp.bfloat16) else accum_dtype
        ),
    )

    return out.astype(q.dtype)


def attention_multi_chunk(
    q: Float[Array, "batch seq heads head_dim"],
    k: Float[Array, "batch seq heads head_dim"],
    v: Float[Array, "batch seq heads value_dim"],
    chunk_size: int,
    start_index: Int[Array, ""],
    rotary: RotaryEmbedding,
    mask: Bool[Array, "batch seq"] | None = None,
    segment_ids: Int[Array, "batch seq"] | None = None,
    position_ids: Int[Array, "batch seq"] | None = None,
    accum_dtype: jnp.dtype = jnp.float32,
    softmax_dtype: jnp.dtype = jnp.float32,
    dropout_rate: float = 0.0,
    dropout_mode: AttentionDropoutMode = "post_softmax",
    deterministic: bool = True,
    key: PRNGKeyArray | None = None,
    *,
    _segment_metadata: SegmentMetadata | None = None,
    _rotary_table: RotaryTable | None = None,
) -> Float[Array, "batch seq heads value_dim"]:
    """Block-diagonal chunked attention for training.

    Processes sequences in fixed-size chunks with independent attention
    per chunk. This enforces the block-diagonal structure where attention
    only operates within chunks - cross-chunk context flows through CEMA.

    :param jax.Array q: Query tensor of shape (batch, seq, heads, head_dim).
    :param jax.Array k: Key tensor of shape (batch, seq, heads, head_dim).
    :param jax.Array v: Value tensor of shape (batch, seq, heads, value_dim).
    :param int chunk_size: Size of each attention chunk.
    :param jax.Array start_index: Absolute position offset for RoPE (JAX scalar).
    :param RotaryEmbedding rotary: RotaryEmbedding module for position encoding.
    :param jax.Array | None mask: Optional mask where True marks valid positions.
    :param jax.Array | None segment_ids: Optional segment IDs for strict attention blocking.
    :param jax.Array | None position_ids: Optional explicit per-token positions for RoPE.
        When omitted with segment_ids given, per-document positions (restarting
        at each segment start) are derived automatically.
    :param jnp.dtype accum_dtype: Accumulation dtype for attention matmuls.
    :param jnp.dtype softmax_dtype: Dtype used to evaluate attention softmax.
    :param float dropout_rate: Attention dropout rate.
    :param AttentionDropoutMode dropout_mode: Post-softmax dropout or DropKey.
    :param bool deterministic: If True, skip dropout.
    :param PRNGKeyArray | None key: PRNG key for dropout.
    :param SegmentMetadata | None _segment_metadata: Internal per-model-call metadata.
    :param RotaryTable | None _rotary_table: Internal per-model-call rotary factors.
    :return jax.Array: Output tensor of shape (batch, seq, heads, value_dim).
    """
    B, L, H, Dh = q.shape
    Dv = v.shape[-1]

    # Handle empty or very short sequences
    if L == 0:
        return jnp.zeros((B, L, H, Dv), dtype=q.dtype)

    if segment_ids is not None and _segment_metadata is None:
        _segment_metadata = derive_segment_metadata(segment_ids)

    if _rotary_table is None:
        if segment_ids is not None and position_ids is None:
            # RoPE must restart at each packed document: without explicit
            # position_ids a packed row would silently keep continuous global
            # phases across boundaries while the mask still isolates attention.
            assert _segment_metadata is not None
            position_ids = _segment_metadata.local_positions

        if position_ids is None:
            local_positions = (
                jnp.arange(L, dtype=jnp.int32) + start_index.astype(jnp.int32)
            ) % chunk_size
            position_ids = jnp.broadcast_to(local_positions, (B, L))
        else:
            position_ids = position_ids % chunk_size

    if L <= chunk_size:
        # Single-chunk fast path avoids padding/chunk reshapes. Keep its RoPE,
        # mask, and dropout semantics in lockstep with the multi-chunk path.
        q_rot, k_rot = rotary(
            q,
            k,
            start_index,
            position_ids=position_ids,
            table=_rotary_table,
        )
        qk_mask = None
        if segment_ids is not None:
            # Compare contiguous runs (ids may repeat); validity from raw ids.
            # Local chunks are all 0 here (L <= chunk_size), so run equality
            # alone gives the full same-run, same-local-chunk condition.
            assert _segment_metadata is not None
            seg_runs = _segment_metadata.run_ids
            segment_valid = _segment_metadata.valid
            qk_mask = (
                (seg_runs[:, :, None] == seg_runs[:, None, :])
                & segment_valid[:, :, None]
                & segment_valid[:, None, :]
            )
        return attention_single_chunk(
            q_rot,
            k_rot,
            v,
            kv_mask=mask,
            qk_mask=qk_mask,
            accum_dtype=accum_dtype,
            softmax_dtype=softmax_dtype,
            causal=True,
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode,
            deterministic=deterministic,
            key=key,
        )

    # Pad to multiple of chunk_size
    pad_len = (chunk_size - L % chunk_size) % chunk_size
    if pad_len > 0:
        q = jnp.pad(q, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        k = jnp.pad(k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        v = jnp.pad(v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
        if mask is not None:
            mask = jnp.pad(mask, ((0, 0), (0, pad_len)), constant_values=False)
        if segment_ids is not None:
            segment_ids = jnp.pad(segment_ids, ((0, 0), (0, pad_len)), constant_values=0)
        if position_ids is not None:
            position_ids = jnp.pad(position_ids, ((0, 0), (0, pad_len)), constant_values=0)
        if _segment_metadata is not None:
            _segment_metadata = SegmentMetadata(
                valid=jnp.pad(
                    _segment_metadata.valid,
                    ((0, 0), (0, pad_len)),
                    constant_values=False,
                ),
                boundaries=jnp.pad(
                    _segment_metadata.boundaries,
                    ((0, 0), (0, pad_len)),
                    constant_values=False,
                ),
                run_ids=jnp.pad(
                    _segment_metadata.run_ids,
                    ((0, 0), (0, pad_len)),
                    constant_values=0,
                ),
                local_positions=jnp.pad(
                    _segment_metadata.local_positions,
                    ((0, 0), (0, pad_len)),
                    constant_values=0,
                ),
            )
        if _rotary_table is not None:
            _rotary_table = RotaryTable(
                cos=jnp.pad(
                    _rotary_table.cos,
                    ((0, 0), (0, pad_len), (0, 0), (0, 0)),
                    constant_values=1.0,
                ),
                sin=jnp.pad(
                    _rotary_table.sin,
                    ((0, 0), (0, pad_len), (0, 0), (0, 0)),
                    constant_values=0.0,
                ),
            )

    L_padded = q.shape[1]
    num_chunks = L_padded // chunk_size

    # Apply RoPE once with absolute positions, then reshape into chunks.
    q_rot_full, k_rot_full = rotary(
        q,
        k,
        start_index,
        position_ids=position_ids,
        table=_rotary_table,
    )
    q_rot = q_rot_full.reshape(B * num_chunks, chunk_size, H, Dh)
    k_rot = k_rot_full.reshape(B * num_chunks, chunk_size, H, Dh)
    v_chunked = v.reshape(B, num_chunks, chunk_size, H, Dv).reshape(
        B * num_chunks, chunk_size, H, Dv
    )

    mask_chunked = None
    if mask is not None:
        mask_chunked = mask.reshape(B * num_chunks, chunk_size)

    if segment_ids is not None:
        # Chunk boundaries are re-anchored at each segment start so a packed
        # document attends exactly as it would running alone, instead of
        # being split at global chunk-grid offsets. A re-anchored chunk spans
        # at most chunk_size consecutive positions, so keys/values from the
        # current + previous global chunk cover every allowed edge; the mask
        # keeps only same-run, same-local-chunk, valid pairs.
        assert _segment_metadata is not None
        run_ids = _segment_metadata.run_ids
        local_positions = _segment_metadata.local_positions
        local_chunks = local_positions // chunk_size

        def window_keys(z: jnp.ndarray, fill_value: jax.typing.ArrayLike) -> jnp.ndarray:
            """Prepend each chunk's predecessor along the key axis.

            :param jnp.ndarray z: Per-token array of shape (B, L_padded, ...).
            :param jax.typing.ArrayLike fill_value: Fill for the first chunk's missing predecessor.
            :return jnp.ndarray: Windowed array of shape (B*NC, 2*chunk, ...).
            """
            zc = z.reshape(B, num_chunks, chunk_size, *z.shape[2:])
            prev = jnp.concatenate([jnp.full_like(zc[:, :1], fill_value), zc[:, :-1]], axis=1)
            return jnp.concatenate([prev, zc], axis=2).reshape(
                B * num_chunks, 2 * chunk_size, *z.shape[2:]
            )

        runs_q = run_ids.reshape(B * num_chunks, chunk_size)
        locals_q = local_chunks.reshape(B * num_chunks, chunk_size)
        seg_q = segment_ids.reshape(B * num_chunks, chunk_size)
        # Run ids start at 1, so fill 0 can never match a real run
        runs_k = window_keys(run_ids, 0)
        locals_k = window_keys(local_chunks, 0)
        seg_k = window_keys(segment_ids, 0)
        qk_mask_windowed = (
            (runs_q[:, :, None] == runs_k[:, None, :])
            & (locals_q[:, :, None] == locals_k[:, None, :])
            & (seg_q[:, :, None] > 0)
            & (seg_k[:, None, :] > 0)
        )

        # causal=True with L_q < L_kv gives k_pos <= q_pos + chunk_size,
        # which is exactly global causality for the [prev, current] window
        out_chunked = attention_single_chunk(
            q_rot,
            window_keys(k_rot_full, 0.0),
            window_keys(v, 0.0),
            kv_mask=window_keys(mask, False) if mask is not None else None,
            qk_mask=qk_mask_windowed,
            accum_dtype=accum_dtype,
            softmax_dtype=softmax_dtype,
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode,
            deterministic=deterministic,
            key=key,
        )
    else:
        out_chunked = attention_single_chunk(
            q_rot,
            k_rot,
            v_chunked,
            kv_mask=mask_chunked,
            qk_mask=None,
            accum_dtype=accum_dtype,
            softmax_dtype=softmax_dtype,
            dropout_rate=dropout_rate,
            dropout_mode=dropout_mode,
            deterministic=deterministic,
            key=key,
        )

    # Reshape back: (B * num_chunks, chunk_size, H, Dv) -> (B, L_padded, H, Dv)
    out = out_chunked.reshape(B, L_padded, H, Dv)

    # Remove padding
    if pad_len > 0:
        out = out[:, :L, :, :]

    return out


# -----------------------------------------------------------------------------
# ChunkedAttention Module
# -----------------------------------------------------------------------------


class ChunkedAttention(eqx.Module):
    """Inner RoPE attention with exact chunk-local or sliding-window caching.

    The released mode uses independent chunks and local rotary positions.
    Setting attention_window enables an intentional sliding-window extension
    with absolute rotary positions. Both modes use a fixed-capacity ring whose
    per-query validity is derived from token ages, so outputs do not depend on
    how calls are partitioned.
    """

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    value_head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    attention_window: int | None = eqx.field(static=True)
    attention_dropout: float = eqx.field(static=True)
    attention_dropout_mode: AttentionDropoutMode = eqx.field(static=True)
    accum_dtype: jnp.dtype = eqx.field(static=True)
    softmax_dtype: jnp.dtype = eqx.field(static=True)
    rotary: RotaryEmbedding

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        rope_base: float = 10000.0,
        attention_window: int | None = None,
        attention_dropout: float = 0.0,
        attention_dropout_mode: AttentionDropoutMode = "post_softmax",
        accum_dtype: jnp.dtype = jnp.float32,
        softmax_dtype: jnp.dtype = jnp.float32,
    ):
        """Initialize chunked attention.

        :param int num_heads: Number of attention heads.
        :param int head_dim: Query/key width per head.
        :param int value_head_dim: Value width per head.
        :param int chunk_size: Released local-attention chunk size.
        :param float rope_base: Rotary frequency base.
        :param int | None attention_window: Optional sliding-window width.
        :param float attention_dropout: Attention dropout probability.
        :param AttentionDropoutMode attention_dropout_mode: Dropout placement.
        :param jnp.dtype accum_dtype: Attention matmul accumulation dtype.
        :param jnp.dtype softmax_dtype: Attention softmax dtype.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if attention_window is not None and attention_window <= 0:
            raise ValueError(
                f"attention_window must be positive when provided, got {attention_window}"
            )
        _validate_dropout_rate("attention_dropout", attention_dropout)
        _validate_dropout_mode(attention_dropout_mode)
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.chunk_size = chunk_size
        self.attention_window = attention_window
        self.attention_dropout = attention_dropout
        self.attention_dropout_mode = attention_dropout_mode
        self.accum_dtype = accum_dtype
        self.softmax_dtype = softmax_dtype
        self.rotary = RotaryEmbedding(dim=head_dim, base=rope_base)

    @property
    def cache_capacity(self) -> int:
        """Return the fixed ring-buffer width."""
        return self.chunk_size if self.attention_window is None else self.attention_window

    def _full_sliding_attention(
        self,
        q: Float[Array, "batch seq heads head_dim"],
        k: Float[Array, "batch seq heads head_dim"],
        v: Float[Array, "batch seq heads value_dim"],
        mask: Bool[Array, "batch seq"] | None,
        segment_ids: Int[Array, "batch seq"] | None,
        position_ids: Int[Array, "batch seq"] | None,
        deterministic: bool,
        key: PRNGKeyArray | None,
        segment_metadata: SegmentMetadata | None = None,
        rotary_table: RotaryTable | None = None,
    ) -> Float[Array, "batch seq heads value_dim"]:
        """Evaluate the optional sliding extension without a cache.

        :param Float[Array, "batch seq heads head_dim"] q: Query vectors.
        :param Float[Array, "batch seq heads head_dim"] k: Key vectors.
        :param Float[Array, "batch seq heads value_dim"] v: Value vectors.
        :param Bool[Array, "batch seq"] | None mask: Optional token-validity mask.
        :param Int[Array, "batch seq"] | None segment_ids: Optional packed-sequence IDs.
        :param Int[Array, "batch seq"] | None position_ids: Optional rotary positions.
        :param bool deterministic: Whether attention dropout is disabled.
        :param PRNGKeyArray | None key: Random key used by attention dropout.
        :param SegmentMetadata | None segment_metadata: Optional shared packed metadata.
        :param RotaryTable | None rotary_table: Optional shared rotary factors.
        :return Float[Array, "batch seq heads value_dim"]: Sliding-attention output.
        """
        batch, length = q.shape[:2]
        if segment_ids is not None:
            if segment_metadata is None:
                segment_metadata = derive_segment_metadata(segment_ids)
            run_ids = segment_metadata.run_ids
            local_positions = segment_metadata.local_positions
            rotary_positions = (
                None
                if rotary_table is not None
                else (local_positions if position_ids is None else position_ids)
            )
            token_positions = local_positions
            same_run = run_ids[:, :, None] == run_ids[:, None, :]
            segment_valid = segment_metadata.valid
            valid_segments = segment_valid[:, :, None] & segment_valid[:, None, :]
        else:
            token_positions = jnp.broadcast_to(jnp.arange(length, dtype=jnp.int32), (batch, length))
            rotary_positions = (
                None
                if rotary_table is not None
                else (token_positions if position_ids is None else position_ids)
            )
            same_run = jnp.ones((batch, length, length), dtype=jnp.bool_)
            valid_segments = same_run

        age = token_positions[:, :, None] - token_positions[:, None, :]
        qk_mask = same_run & valid_segments & (age >= 0) & (age < self.attention_window)
        if mask is not None:
            qk_mask = qk_mask & mask[:, :, None] & mask[:, None, :]
        q_rot, k_rot = self.rotary(
            q,
            k,
            jnp.asarray(0, dtype=jnp.int32),
            position_ids=rotary_positions,
            table=rotary_table,
        )
        return attention_single_chunk(
            q_rot,
            k_rot,
            v,
            qk_mask=qk_mask,
            accum_dtype=self.accum_dtype,
            softmax_dtype=self.softmax_dtype,
            causal=False,
            dropout_rate=self.attention_dropout,
            dropout_mode=self.attention_dropout_mode,
            deterministic=deterministic,
            key=key,
        )

    def _prefill(
        self,
        q: Float[Array, "batch seq heads head_dim"],
        k: Float[Array, "batch seq heads head_dim"],
        v: Float[Array, "batch seq heads value_dim"],
        deterministic: bool,
        key: PRNGKeyArray | None,
        rotary_table: RotaryTable | None = None,
    ) -> tuple[
        Float[Array, "batch seq heads value_dim"],
        AttentionCache,
        Int[Array, ""],
    ]:
        """Evaluate a pristine prompt vectorially and materialize its ring tail.

        :param Float[Array, "batch seq heads head_dim"] q: Prompt query vectors.
        :param Float[Array, "batch seq heads head_dim"] k: Prompt key vectors.
        :param Float[Array, "batch seq heads value_dim"] v: Prompt value vectors.
        :param bool deterministic: Whether attention dropout is disabled.
        :param PRNGKeyArray | None key: Random key used by attention dropout.
        :param RotaryTable | None rotary_table: Optional shared rotary factors.
        :return tuple: Attention output, materialized ring cache, and prompt length.
        """
        batch, length = q.shape[:2]
        if self.attention_window is None:
            out = attention_multi_chunk(
                q,
                k,
                v,
                chunk_size=self.chunk_size,
                start_index=jnp.asarray(0, dtype=jnp.int32),
                rotary=self.rotary,
                accum_dtype=self.accum_dtype,
                softmax_dtype=self.softmax_dtype,
                dropout_rate=self.attention_dropout,
                dropout_mode=self.attention_dropout_mode,
                deterministic=deterministic,
                key=key,
                _rotary_table=rotary_table,
            )
        else:
            out = self._full_sliding_attention(
                q,
                k,
                v,
                None,
                None,
                None,
                deterministic,
                key,
                rotary_table=rotary_table,
            )

        capacity = self.cache_capacity
        keep = min(length, capacity)
        tail_start = length - keep
        absolute_times = jnp.arange(tail_start, length, dtype=jnp.int32)
        rope_positions = (
            absolute_times % self.chunk_size if self.attention_window is None else absolute_times
        )

        # Recompute only the retained tail instead of retaining all rotated keys
        # from vectorized attention. A model-level table avoids repeating trig;
        # standalone calls derive the same bounded tail table here.
        tail_table = (
            RotaryTable(
                cos=rotary_table.cos[:, tail_start:],
                sin=rotary_table.sin[:, tail_start:],
            )
            if rotary_table is not None
            else self.rotary.table_from_positions(rope_positions)
        )
        _, k_tail_rot = self.rotary(
            q[:, tail_start:],
            k[:, tail_start:],
            jnp.asarray(0, dtype=jnp.int32),
            table=tail_table,
        )
        slots = absolute_times % capacity
        cache_k = (
            jnp
            .zeros(
                (batch, capacity, self.num_heads, self.head_dim),
                dtype=k.dtype,
            )
            .at[:, slots]
            .set(k_tail_rot)
        )
        cache_v = (
            jnp
            .zeros(
                (batch, capacity, self.num_heads, self.value_head_dim),
                dtype=v.dtype,
            )
            .at[:, slots]
            .set(v[:, tail_start:])
        )
        count = jnp.asarray(length, dtype=jnp.int32)
        return out, AttentionCache(k=cache_k, v=cache_v, count=count), count

    def __call__(
        self,
        q: Float[Array, "batch seq heads head_dim"],
        k: Float[Array, "batch seq heads head_dim"],
        v: Float[Array, "batch seq heads value_dim"],
        cache: AttentionCache | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
        *,
        _cache_validated: bool = False,
        _segment_metadata: SegmentMetadata | None = None,
        _rotary_table: RotaryTable | None = None,
    ) -> tuple[
        Float[Array, "batch seq heads value_dim"],
        AttentionCache | None,
        Int[Array, ""],
    ]:
        """Apply attention and optionally return continuation state."""
        if q.ndim != 4 or k.shape != q.shape:
            raise ValueError(f"q and k must share rank-4 shape, got {q.shape} and {k.shape}")
        if v.shape[:3] != q.shape[:3]:
            raise ValueError(f"v prefix shape must match q, got {v.shape} and {q.shape}")
        batch, length, heads, _ = q.shape
        if heads != self.num_heads:
            raise ValueError(f"expected {self.num_heads} heads, got {heads}")
        streaming = cache is not None or return_cache
        if streaming and (segment_ids is not None or position_ids is not None):
            raise ValueError("segment_ids/position_ids are unsupported with cached attention")
        if streaming and mask is not None:
            raise ValueError("attention masks are unsupported with cached attention")
        if not deterministic and self.attention_dropout > 0.0 and key is None:
            raise ValueError("PRNG key required for attention dropout")

        if not streaming:
            if self.attention_window is None:
                out = attention_multi_chunk(
                    q,
                    k,
                    v,
                    chunk_size=self.chunk_size,
                    start_index=jnp.asarray(0, dtype=jnp.int32),
                    rotary=self.rotary,
                    mask=mask,
                    segment_ids=segment_ids,
                    position_ids=position_ids,
                    accum_dtype=self.accum_dtype,
                    softmax_dtype=self.softmax_dtype,
                    dropout_rate=self.attention_dropout,
                    dropout_mode=self.attention_dropout_mode,
                    deterministic=deterministic,
                    key=key,
                    _segment_metadata=_segment_metadata,
                    _rotary_table=_rotary_table,
                )
            else:
                out = self._full_sliding_attention(
                    q,
                    k,
                    v,
                    mask,
                    segment_ids,
                    position_ids,
                    deterministic,
                    key,
                    segment_metadata=_segment_metadata,
                    rotary_table=_rotary_table,
                )
            return out, None, jnp.asarray(length, dtype=jnp.int32)

        # Pristine prefill has no history-dependent input. When attention
        # dropout is inactive, its outputs are the normal vectorized result;
        # only the fixed-capacity ring state needs to be materialized.
        attention_dropout_active = not deterministic and self.attention_dropout > 0.0
        if cache is None and return_cache and not attention_dropout_active and length > 0:
            return self._prefill(q, k, v, deterministic, key, _rotary_table)

        capacity = self.cache_capacity
        if cache is None:
            cache_k = jnp.zeros(
                (batch, capacity, self.num_heads, self.head_dim),
                dtype=q.dtype,
            )
            cache_v = jnp.zeros(
                (batch, capacity, self.num_heads, self.value_head_dim),
                dtype=v.dtype,
            )
            count = jnp.asarray(0, dtype=jnp.int32)
        else:
            expected_k = (batch, capacity, self.num_heads, self.head_dim)
            expected_v = (batch, capacity, self.num_heads, self.value_head_dim)
            if cache.k.shape != expected_k or cache.v.shape != expected_v:
                raise ValueError(
                    f"cache shapes must be {expected_k} and {expected_v}, "
                    f"got {cache.k.shape} and {cache.v.shape}"
                )
            cache_k = cache.k
            cache_v = cache.v
            count = cache.count.astype(jnp.int32)

        if length == 0:
            new_cache = AttentionCache(k=cache_k, v=cache_v, count=count) if return_cache else None
            return v, new_cache, count

        if cache is not None and not _cache_validated:
            count = eqx.error_if(
                count,
                (count < 0) | (count > jnp.iinfo(jnp.int32).max - length),
                "attention cache count must be non-negative and must not overflow int32",
            )

        use_rng = not deterministic and self.attention_dropout > 0.0
        rng = key if key is not None else jax.random.PRNGKey(0)
        slots = jnp.arange(capacity, dtype=jnp.int32)

        def step(
            carry: tuple[Array, Array, Array, Array],
            inputs: tuple[Array, ...],
        ) -> tuple[tuple[Array, Array, Array, Array], Array]:
            """Process one token and advance the ring-buffer state.

            :param tuple[Array, Array, Array, Array] carry: Cached keys, cached values,
                absolute position, and dropout RNG state.
            :param tuple[Array, ...] inputs: Query, key, value, and optional rotary factors.
            :return tuple[tuple[Array, Array, Array, Array], Array]: Updated ring-buffer carry
                and the token's attention output.
            """
            current_k, current_v, position, current_rng = carry
            q_t, k_t, v_t = inputs[:3]
            if _rotary_table is None:
                rope_position = (
                    position % self.chunk_size if self.attention_window is None else position
                )
                q_rot, k_rot = self.rotary(
                    q_t[:, None],
                    k_t[:, None],
                    rope_position,
                )
            else:
                cos_t, sin_t = inputs[3:]
                q_rot, k_rot = self.rotary(
                    q_t[:, None],
                    k_t[:, None],
                    position,
                    table=RotaryTable(
                        cos=cos_t[:, None, :, :],
                        sin=sin_t[:, None, :, :],
                    ),
                )
            write_slot = position % capacity
            current_k = current_k.at[:, write_slot].set(k_rot[:, 0])
            current_v = current_v.at[:, write_slot].set(v_t)

            slot_time = position - jnp.mod(position - slots, capacity)
            valid = slot_time >= 0
            if self.attention_window is None:
                valid = valid & (slot_time // self.chunk_size == position // self.chunk_size)
            kv_mask = jnp.broadcast_to(valid, (batch, capacity))

            if use_rng:
                current_rng, step_key = jax.random.split(current_rng)
            else:
                step_key = None
            output = attention_single_chunk(
                q_rot,
                current_k,
                current_v,
                kv_mask=kv_mask,
                accum_dtype=self.accum_dtype,
                softmax_dtype=self.softmax_dtype,
                causal=False,
                dropout_rate=self.attention_dropout,
                dropout_mode=self.attention_dropout_mode,
                deterministic=deterministic,
                key=step_key,
            )
            return (
                current_k,
                current_v,
                position + 1,
                current_rng,
            ), output[:, 0]

        scan_inputs: tuple[Array, ...] = (
            jnp.swapaxes(q, 0, 1),
            jnp.swapaxes(k, 0, 1),
            jnp.swapaxes(v, 0, 1),
        )
        if _rotary_table is not None:
            scan_inputs = scan_inputs + (
                jnp.moveaxis(_rotary_table.cos, 1, 0),
                jnp.moveaxis(_rotary_table.sin, 1, 0),
            )
        (final_k, final_v, final_count, _), outputs = jax.lax.scan(
            step,
            (cache_k, cache_v, count, rng),
            scan_inputs,
        )
        out = jnp.swapaxes(outputs, 0, 1)
        new_cache = (
            AttentionCache(k=final_k, v=final_v, count=final_count) if return_cache else None
        )
        return out, new_cache, final_count


# -----------------------------------------------------------------------------
# Complete Megalodon Attention Block
# -----------------------------------------------------------------------------


class MegalodonAttention(eqx.Module):
    """Complete Megalodon attention block with CEMA and gating.

    Architecture flow:
        x → TimestepNorm → x_tn
                    ↓
               CEMA(x_tn.T).T → RMSNorm → mx
                    ↓
        ┌──────────┼────────────────────────────────┐
        ↓          ↓                                ↓
      wz(mx)    wv(x_tn)                         wr(mx)
        ↓          ↓                                ↓
    split heads  SiLU                             SiLU
        ↓          ↓                                ↓
    per-head      V                              r (gate)
    RMS norm
        ↓
    affine → Q, K
        ↓
    attention(Q, K, V) → out
        ↓
    out * r → wh2
        ↓
    wh1(mx) + wh2(out*r) → h
        ↓
    h + x (residual) → y
    """

    # Sub-modules
    timenorm: TimestepNorm
    cema: ComplexEMA
    rmsnorm: RMSNorm
    inner: ChunkedAttention

    # Projections
    wz: eqx.nn.Linear  # D → z_dim (shared Q/K)
    wv: eqx.nn.Linear  # D → value_dim
    wr: eqx.nn.Linear  # D → value_dim (gate)
    wh1: eqx.nn.Linear  # D → D
    wh2: eqx.nn.Linear  # value_dim → D

    # Per-head affine for Q/K (gamma+1 parameterization)
    gamma: Float[Array, "2 z_dim"]
    beta: Float[Array, "2 z_dim"]

    # Config
    model_dim: int = eqx.field(static=True)
    value_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    value_head_dim: int = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    hidden_dropout: float = eqx.field(static=True)
    compute_dtype: jnp.dtype = eqx.field(static=True)
    accum_dtype: jnp.dtype = eqx.field(static=True)
    use_associative_segment_scan: bool = eqx.field(static=True)

    def __init__(
        self,
        model_dim: int,
        z_dim: int,
        value_dim: int,
        num_heads: int,
        cema_ndim: int,
        chunk_size: int,
        norm_num_groups: int,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        rope_base: float = 10000.0,
        attention_window: int | None = None,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        attention_dropout_mode: AttentionDropoutMode = "post_softmax",
        hidden_dropout: float = 0.0,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        accum_dtype: jnp.dtype = jnp.float32,
        attention_softmax_dtype: jnp.dtype = jnp.float32,
        use_associative_segment_scan: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize MegalodonAttention.

        :param int model_dim: Model hidden dimension D.
        :param int z_dim: Shared Q/K dimension.
        :param int value_dim: Value dimension.
        :param int num_heads: Number of attention heads.
        :param int cema_ndim: Number of EMA orders for ComplexEMA.
        :param int chunk_size: Attention chunk size.
        :param int norm_num_groups: Number of groups for TimestepNorm.
        :param float norm_eps: Epsilon for normalization.
        :param bool norm_affine: Whether RMSNorm includes an affine scale. TimestepNorm is always
            affine for released-source compatibility.
        :param float rope_base: Base for rotary embeddings.
        :param int | None attention_window: Optional sliding-window width.
        :param float dropout: Output dropout rate.
        :param float attention_dropout: Attention weight dropout rate.
        :param AttentionDropoutMode attention_dropout_mode: Post-softmax dropout or DropKey.
        :param float hidden_dropout: Hidden layer dropout rate.
        :param jnp.dtype param_dtype: Parameter storage dtype for Linear weights.
        :param jnp.dtype compute_dtype: Compute dtype for matmuls and activations.
        :param jnp.dtype accum_dtype: Accumulation dtype for matmuls/reductions.
        :param jnp.dtype attention_softmax_dtype: Attention softmax evaluation dtype.
        :param bool use_associative_segment_scan: Segmented CEMA implementation for
            packed sequences: parallel associative scan (default) or the
            sequential low-memory fallback.
        :param PRNGKeyArray key: PRNG key for initialization.
        """
        _validate_dropout_rate("dropout", dropout)
        _validate_dropout_rate("attention_dropout", attention_dropout)
        _validate_dropout_rate("hidden_dropout", hidden_dropout)
        self.model_dim = model_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = z_dim // num_heads
        self.value_head_dim = value_dim // num_heads
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout
        self.compute_dtype = compute_dtype
        self.accum_dtype = accum_dtype
        self.use_associative_segment_scan = use_associative_segment_scan

        # Split keys
        keys = jax.random.split(key, 9)

        # Sub-modules
        self.timenorm = TimestepNorm(
            num_features=model_dim,
            num_groups=norm_num_groups,
            eps=norm_eps,
        )
        self.cema = ComplexEMA(embed_dim=model_dim, ndim=cema_ndim, key=keys[0])
        self.rmsnorm = RMSNorm(dim=model_dim, eps=norm_eps, affine=norm_affine)
        self.inner = ChunkedAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            value_head_dim=self.value_head_dim,
            chunk_size=chunk_size,
            rope_base=rope_base,
            attention_window=attention_window,
            attention_dropout=attention_dropout,
            attention_dropout_mode=attention_dropout_mode,
            accum_dtype=accum_dtype,
            softmax_dtype=attention_softmax_dtype,
        )

        # Projections
        self.wz = eqx.nn.Linear(model_dim, z_dim, dtype=param_dtype, key=keys[2])
        self.wv = eqx.nn.Linear(model_dim, value_dim, dtype=param_dtype, key=keys[3])
        self.wr = eqx.nn.Linear(model_dim, value_dim, dtype=param_dtype, key=keys[4])
        self.wh1 = eqx.nn.Linear(model_dim, model_dim, dtype=param_dtype, key=keys[5])
        self.wh2 = eqx.nn.Linear(
            value_dim,
            model_dim,
            use_bias=False,
            dtype=param_dtype,
            key=keys[6],
        )

        # Per-head affine parameters (gamma+1 parameterization, init zeros)
        self.gamma = jnp.zeros((2, z_dim), dtype=jnp.float32)
        self.beta = jnp.zeros((2, z_dim), dtype=jnp.float32)

    def __call__(
        self,
        x: Float[Array, "batch seq dim"],
        cache: LayerCache | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        position_ids: Int[Array, "batch seq"] | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
        *,
        _cache_validated: bool = False,
        _segment_metadata: SegmentMetadata | None = None,
        _rotary_table: RotaryTable | None = None,
    ) -> tuple[Float[Array, "batch seq dim"], LayerCache | None]:
        """Forward pass through the attention block.

        :param jax.Array x: Input tensor of shape (batch, seq, dim).
        :param LayerCache | None cache: Optional layer cache from previous tokens.
        :param jax.Array | None mask: Optional mask where True marks valid positions.
        :param jax.Array | None segment_ids: Optional segment IDs for strict attention blocking.
        :param jax.Array | None position_ids: Optional explicit per-token positions for RoPE.
            When omitted with segment_ids given, document-local positions are derived.
        :param bool return_cache: Whether to return updated cache.
        :param bool deterministic: If True, skip dropout.
        :param PRNGKeyArray | None key: PRNG key for dropout.
        :param bool _cache_validated: Internal signal that model cache counts were checked.
        :param SegmentMetadata | None _segment_metadata: Internal per-model-call metadata.
        :param RotaryTable | None _rotary_table: Internal per-model-call rotary factors.
        :return tuple[jax.Array, LayerCache | None]: Output tensor and updated cache.
        """
        if (
            not deterministic
            and key is None
            and any(
                rate > 0.0
                for rate in (self.dropout, self.inner.attention_dropout, self.hidden_dropout)
            )
        ):
            raise ValueError("PRNG key required when deterministic=False and dropout is enabled")
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim
        Dv = self.value_head_dim

        if cache is not None and not _cache_validated:
            x = eqx.error_if(
                x,
                layer_cache_invariant_violation(
                    cache,
                    batch_size=B,
                    cache_capacity=self.inner.cache_capacity,
                    num_heads=H,
                    head_dim=Dh,
                    value_head_dim=Dv,
                    model_dim=self.model_dim,
                    cema_ndim=self.cema.ndim,
                    norm_num_groups=self.timenorm.num_groups,
                    compute_dtype=self.compute_dtype,
                    increment=L,
                ),
                CACHE_INVARIANT_MESSAGE,
            )
            if cache.attn is None and cache.norm is None and cache.ema is None:
                # Canonicalize the sparse zero-history public representation.
                cache = None
            _cache_validated = True

        x = x.astype(self.compute_dtype)

        # Split keys for dropout (4 keys: attention, mx_dropout, gated_dropout, output_dropout)
        if key is not None:
            k1, k2, k3, k4 = jax.random.split(key, 4)
        else:
            k1 = k2 = k3 = k4 = None

        # Extract cache components
        norm_state = cache.norm if cache is not None else None
        ema_state = cache.ema.h if cache is not None and cache.ema is not None else None
        attn_cache = cache.attn if cache is not None else None

        # TimestepNorm (segment_ids resets running stats at packed-doc boundaries)
        x_tn, new_norm_state = self.timenorm(
            x,
            state=norm_state,
            mask=mask,
            segment_ids=segment_ids,
            _state_validated=_cache_validated,
            _segment_metadata=_segment_metadata,
        )

        # CEMA: (B, L, D) -> (B, D, L) -> CEMA -> (B, D, L) -> (B, L, D)
        # Pass mask to prevent EMA state contamination from padded positions;
        # segment_ids resets the EMA state at packed-doc boundaries
        need_ema_state = return_cache or ema_state is not None
        cema_input = x_tn.transpose(0, 2, 1)  # (B, D, L)

        y_cema, h_last = self.cema(
            cema_input,
            h_init=ema_state,
            return_state=need_ema_state,
            mask=mask,
            segment_ids=segment_ids,
            use_associative_segment_scan=self.use_associative_segment_scan,
            _segment_metadata=_segment_metadata,
        )
        y_cema = y_cema.transpose(0, 2, 1)  # (B, L, D)

        # RMSNorm on CEMA output, then hidden_dropout (matching PyTorch reference line 1370)
        mx = self.rmsnorm(y_cema)
        if not deterministic and self.hidden_dropout > 0.0:
            mx = inverted_dropout(mx, self.hidden_dropout, k2)

        # Shared Z projection for Q/K
        z = linear_3d(self.wz, mx, self.compute_dtype, self.accum_dtype)  # (B, L, z_dim)
        z = z.reshape(B, L, H, Dh)

        # Per-head RMS normalization (in fp32 for stability)
        z_normed = _rms_normalize(z, self.norm_eps)

        # Affine transform to Q and K in fp32 for stability
        # gamma, beta: (2, z_dim) -> (2, H, Dh)
        gamma_heads = self.gamma.reshape(2, H, Dh).astype(jnp.float32)
        beta_heads = self.beta.reshape(2, H, Dh).astype(jnp.float32)
        scale = (gamma_heads + 1.0) / jnp.sqrt(jnp.asarray(Dh, dtype=jnp.float32))

        # Broadcast and apply: z_normed is (B, L, H, Dh)
        q = (z_normed * scale[0] + beta_heads[0]).astype(self.compute_dtype)
        k = (z_normed * scale[1] + beta_heads[1]).astype(self.compute_dtype)

        # Value projection with SiLU
        v = jax.nn.silu(
            linear_3d(self.wv, x_tn, self.compute_dtype, self.accum_dtype)
        )  # (B, L, value_dim)
        v = v.reshape(B, L, H, Dv)

        # Gate projection with SiLU
        r = jax.nn.silu(
            linear_3d(self.wr, mx, self.compute_dtype, self.accum_dtype)
        )  # (B, L, value_dim)

        # Inner attention
        if cache is not None and attn_cache is None:
            raise ValueError(CACHE_INVARIANT_MESSAGE)
        out, new_attn_cache, new_position = self.inner(
            q,
            k,
            v,
            cache=attn_cache,
            mask=mask,
            segment_ids=segment_ids,
            position_ids=position_ids,
            return_cache=return_cache,
            deterministic=deterministic,
            key=k1,
            _cache_validated=_cache_validated,
            _segment_metadata=_segment_metadata,
            _rotary_table=_rotary_table,
        )

        # Reshape attention output: (B, L, H, Dv) -> (B, L, value_dim)
        out = out.reshape(B, L, self.value_dim)

        # Apply gate
        gated = out * r

        # Hidden dropout on gated attention output (matching PyTorch reference line 1418)
        if not deterministic and self.hidden_dropout > 0.0:
            gated = inverted_dropout(gated, self.hidden_dropout, k3)

        # Output projections
        h = linear_3d(self.wh1, mx, self.compute_dtype, self.accum_dtype) + linear_3d(
            self.wh2, gated, self.compute_dtype, self.accum_dtype
        )

        # Output dropout
        if not deterministic and self.dropout > 0.0:
            h = inverted_dropout(h, self.dropout, k4)

        # Residual
        y = h + x

        # Build output cache
        if return_cache:
            assert h_last is not None
            new_cache = LayerCache(
                attn=new_attn_cache,
                norm=new_norm_state,
                ema=EMAState(h=h_last),
                position=new_position,
            )
        else:
            new_cache = None

        return y, new_cache


# -----------------------------------------------------------------------------
# NormalizedFFN
# -----------------------------------------------------------------------------


class NormalizedFFN(eqx.Module):
    """Pre-norm Feed-Forward Network with optional SwiGLU.

    Standard path:
        x → LayerNorm → SiLU(fc1) → fc2 → + residual

    SwiGLU path:
        x → LayerNorm → SiLU(fc1) * fc3 → fc2 → + residual

    Supports two-hop residual via residual_base parameter.
    Supports optional residual rescaling (rescale_nffn) for training stability.
    """

    norm: BatchedLayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear | None  # Only for SwiGLU

    swiglu: bool = eqx.field(static=True)
    hidden_dropout: float = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    alpha: Float[Array, "dim"] | None
    compute_dtype: jnp.dtype = eqx.field(static=True)
    accum_dtype: jnp.dtype = eqx.field(static=True)

    def __init__(
        self,
        model_dim: int,
        ffn_hidden_dim: int,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        swiglu: bool = False,
        rescale: bool = False,
        layer_id: int = 0,
        hidden_dropout: float = 0.0,
        dropout: float = 0.0,
        param_dtype: jnp.dtype = jnp.float32,
        compute_dtype: jnp.dtype = jnp.float32,
        accum_dtype: jnp.dtype = jnp.float32,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize NormalizedFFN.

        :param int model_dim: Model hidden dimension.
        :param int ffn_hidden_dim: FFN intermediate dimension.
        :param float norm_eps: Epsilon for layer normalization.
        :param bool norm_affine: Whether the FFN LayerNorm includes affine parameters.
        :param bool swiglu: Whether to use SwiGLU activation.
        :param bool rescale: Whether to apply residual rescaling (rescale_nffn).
        :param int layer_id: Layer index for computing rescale factor (0-indexed).
        :param float hidden_dropout: Dropout after hidden layer.
        :param float dropout: Dropout after output projection.
        :param jnp.dtype param_dtype: Parameter storage dtype for Linear weights.
        :param jnp.dtype compute_dtype: Compute dtype for matmuls and activations.
        :param jnp.dtype accum_dtype: Accumulation dtype for matmuls/reductions.
        :param PRNGKeyArray key: PRNG key for initialization.
        """
        _validate_dropout_rate("dropout", dropout)
        _validate_dropout_rate("hidden_dropout", hidden_dropout)
        self.swiglu = swiglu
        self.hidden_dropout = hidden_dropout
        self.dropout = dropout
        self.compute_dtype = compute_dtype
        self.accum_dtype = accum_dtype
        # Released layer scale is a trainable per-feature vector.
        self.alpha = (
            jnp.full((model_dim,), 0.1 * (0.5**layer_id), dtype=jnp.float32) if rescale else None
        )

        keys = jax.random.split(key, 3)

        self.norm = BatchedLayerNorm(model_dim, eps=norm_eps, affine=norm_affine)
        self.fc1 = eqx.nn.Linear(
            model_dim,
            ffn_hidden_dim,
            use_bias=False,
            dtype=param_dtype,
            key=keys[0],
        )
        self.fc2 = eqx.nn.Linear(
            ffn_hidden_dim,
            model_dim,
            use_bias=False,
            dtype=param_dtype,
            key=keys[1],
        )

        if swiglu:
            self.fc3 = eqx.nn.Linear(
                model_dim,
                ffn_hidden_dim,
                use_bias=False,
                dtype=param_dtype,
                key=keys[2],
            )
        else:
            self.fc3 = None

    def __call__(
        self,
        x: Float[Array, "batch seq dim"],
        residual_base: Float[Array, "batch seq dim"] | None = None,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> Float[Array, "batch seq dim"]:
        """Forward pass through the FFN.

        :param jax.Array x: Input tensor of shape (batch, seq, dim).
        :param jax.Array | None residual_base: Optional tensor for two-hop residuals.
        :param bool deterministic: If True, skip dropout.
        :param PRNGKeyArray | None key: PRNG key for dropout.
        :return jax.Array: Output tensor of shape (batch, seq, dim).
        """
        if not deterministic and key is None and (self.dropout > 0.0 or self.hidden_dropout > 0.0):
            raise ValueError("PRNG key required when deterministic=False and dropout is enabled")
        x = x.astype(self.compute_dtype)
        residual = x if residual_base is None else residual_base.astype(self.compute_dtype)

        # Layer norm (BatchedLayerNorm handles leading dims).
        h = self.norm(x)

        # Hidden layer with activation
        if self.swiglu:
            h = jax.nn.silu(
                linear_3d(self.fc1, h, self.compute_dtype, self.accum_dtype)
            ) * linear_3d(self.fc3, h, self.compute_dtype, self.accum_dtype)
        else:
            h = jax.nn.silu(linear_3d(self.fc1, h, self.compute_dtype, self.accum_dtype))

        # Hidden dropout
        if not deterministic and self.hidden_dropout > 0.0:
            k1, k2 = jax.random.split(key)
            h = inverted_dropout(h, self.hidden_dropout, k1)
        else:
            k2 = key

        # Output projection
        out = linear_3d(self.fc2, h, self.compute_dtype, self.accum_dtype)

        # Output dropout
        if not deterministic and self.dropout > 0.0:
            out = inverted_dropout(out, self.dropout, k2)

        # Apply residual rescaling if enabled (cast to preserve bf16)
        if self.alpha is not None:
            out = out * self.alpha.astype(out.dtype)

        return residual + out
