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

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.config import InitMode
from megalodon_jax.layers.complex_ema import ComplexEMA
from megalodon_jax.layers.norms import RMSNorm
from megalodon_jax.layers.rotary import RotaryEmbedding
from megalodon_jax.layers.timestep_norm import TimestepNorm
from megalodon_jax.types import AttentionCache, EMAState, LayerCache
from megalodon_jax.utils import reinit_linear_weights

# -----------------------------------------------------------------------------
# Attention Primitives (Pure Functions)
# -----------------------------------------------------------------------------


def _linear_3d(
    linear: eqx.nn.Linear, x: Float[Array, "batch seq in_dim"]
) -> Float[Array, "batch seq out_dim"]:
    """Apply an Equinox Linear over a (batch, seq, dim) tensor."""
    y = jnp.matmul(x, linear.weight.T)
    if linear.bias is not None:
        y = y + linear.bias
    return y


def attention_single_chunk(
    q: Float[Array, "batch seq heads head_dim"],
    k: Float[Array, "batch kv_seq heads head_dim"],
    v: Float[Array, "batch kv_seq heads value_dim"],
    kv_mask: Bool[Array, "batch kv_seq"] | None = None,
    causal: bool = True,
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "batch seq heads value_dim"]:
    """Causal attention over a single chunk with NO temperature scaling.

    CRITICAL: This implements normalized attention (scale=1.0), NOT standard
    scaled dot-product attention. The PyTorch reference explicitly sets
    scale=1.0 in F.scaled_dot_product_attention.

    Args:
        q: Query tensor of shape (batch, seq, heads, head_dim).
        k: Key tensor of shape (batch, kv_seq, heads, head_dim).
        v: Value tensor of shape (batch, kv_seq, heads, value_dim).
        kv_mask: Optional key/value mask where True marks valid positions.
        causal: Whether to apply causal masking (default True).
        dropout_rate: Attention dropout rate.
        deterministic: If True, skip dropout.
        key: PRNG key for dropout (required if not deterministic).

    Returns:
        Output tensor of shape (batch, seq, heads, value_dim).
    """
    B, L_q, H, Dh = q.shape
    L_kv = k.shape[1]
    Dv = v.shape[-1]

    # Handle empty sequence
    if L_q == 0 or L_kv == 0:
        return jnp.zeros((B, L_q, H, Dv), dtype=q.dtype)

    # Compute attention scores: (B, H, L_q, L_kv)
    # NO scaling by 1/sqrt(d_k) - this is normalized attention
    # Keep inputs in native dtype but accumulate in float32 for stability/perf.
    scores = jnp.einsum(
        "bqhd,bkhd->bhqk",
        q,
        k,
        preferred_element_type=jnp.float32,
    )

    # Build attention mask
    # Use -inf so softmax(all masked) = NaN, triggering the NaN guard below
    neg_inf = -jnp.inf
    if causal and L_q == L_kv:
        # Standard causal mask for self-attention
        causal_mask = jnp.tril(jnp.ones((L_q, L_kv), dtype=jnp.bool_))
        scores = jnp.where(causal_mask, scores, neg_inf)
    elif causal and L_q < L_kv:
        # Cross-attention with cache: each query attends to all prior keys
        # Query at position i can attend to keys [0, L_kv - L_q + i + 1)
        q_pos = jnp.arange(L_q)[:, None]  # (L_q, 1)
        k_pos = jnp.arange(L_kv)[None, :]  # (1, L_kv)
        # Key position must be <= query position + offset
        offset = L_kv - L_q
        causal_mask = k_pos <= (q_pos + offset)  # (L_q, L_kv)
        scores = jnp.where(causal_mask, scores, neg_inf)

    # Apply key/value padding mask if provided
    if kv_mask is not None:
        # kv_mask: (B, L_kv) with True for valid positions
        kv_mask_expanded = kv_mask[:, None, None, :]  # (B, 1, 1, L_kv)
        scores = jnp.where(kv_mask_expanded, scores, neg_inf)

    # Softmax with NaN guard for fully-masked queries
    # If all keys are masked for a query, softmax(all -inf) = NaN
    # Detect and replace with zeros for safe behavior on padded batches
    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_weights = jnp.where(jnp.isnan(attn_weights), 0.0, attn_weights)

    # Dropout if not deterministic
    if not deterministic and dropout_rate > 0.0:
        if key is None:
            raise ValueError("PRNG key required for dropout")
        keep_mask = jax.random.bernoulli(key, 1.0 - dropout_rate, attn_weights.shape)
        inv_keep = jnp.asarray(1.0 / (1.0 - dropout_rate), dtype=attn_weights.dtype)
        attn_weights = attn_weights * keep_mask.astype(attn_weights.dtype) * inv_keep

    # Apply to values: (B, H, L_q, Dv) -> (B, L_q, H, Dv)
    out = jnp.einsum(
        "bhqk,bkhd->bqhd",
        attn_weights,
        v,
        preferred_element_type=jnp.float32,
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
    dropout_rate: float = 0.0,
    deterministic: bool = True,
    key: PRNGKeyArray | None = None,
) -> Float[Array, "batch seq heads value_dim"]:
    """Block-diagonal chunked attention for training.

    Processes sequences in fixed-size chunks with independent attention
    per chunk. This enforces the block-diagonal structure where attention
    only operates within chunks - cross-chunk context flows through CEMA.

    Args:
        q: Query tensor of shape (batch, seq, heads, head_dim).
        k: Key tensor of shape (batch, seq, heads, head_dim).
        v: Value tensor of shape (batch, seq, heads, value_dim).
        chunk_size: Size of each attention chunk.
        start_index: Absolute position offset for RoPE (JAX scalar).
        rotary: RotaryEmbedding module for position encoding.
        mask: Optional mask where True marks valid positions.
        dropout_rate: Attention dropout rate.
        deterministic: If True, skip dropout.
        key: PRNG key for dropout.

    Returns:
        Output tensor of shape (batch, seq, heads, value_dim).
    """
    B, L, H, Dh = q.shape
    Dv = v.shape[-1]

    # Handle empty or very short sequences
    if L == 0:
        return jnp.zeros((B, L, H, Dv), dtype=q.dtype)

    if L <= chunk_size:
        # Single chunk: apply RoPE and attention directly
        q_rot, k_rot = rotary(q, k, start_index)
        return attention_single_chunk(
            q_rot,
            k_rot,
            v,
            kv_mask=mask,
            causal=True,
            dropout_rate=dropout_rate,
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

    L_padded = q.shape[1]
    num_chunks = L_padded // chunk_size

    # Apply RoPE once with absolute positions, then reshape into chunks.
    q_rot_full, k_rot_full = rotary(q, k, start_index)
    q_rot = q_rot_full.reshape(B * num_chunks, chunk_size, H, Dh)
    k_rot = k_rot_full.reshape(B * num_chunks, chunk_size, H, Dh)
    v_chunked = v.reshape(B, num_chunks, chunk_size, H, Dv).reshape(
        B * num_chunks, chunk_size, H, Dv
    )

    mask_chunked = None
    if mask is not None:
        mask_chunked = mask.reshape(B * num_chunks, chunk_size)

    out_chunked = attention_single_chunk(
        q_rot,
        k_rot,
        v_chunked,
        kv_mask=mask_chunked,
        dropout_rate=dropout_rate,
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
    """Inner attention module with RoPE and optional KV cache.

    Handles both training (multi-chunk) and inference (streaming with cache).

    Cache Behavior:
        The cache behavior is controlled by cache_unbounded and max_cache_len:

        1. cache_unbounded=False, max_cache_len=chunk_size (default):
           "Faithful chunk-local" mode - resets KV at chunk boundaries,
           enforcing block-diagonal attention structure.

        2. cache_unbounded=False, max_cache_len>chunk_size:
           Sliding window mode - no reset at chunk boundaries, keeps a fixed-size
           window of max_cache_len entries. Cross-chunk context is preserved.

        3. cache_unbounded=True:
           No boundary resets, keeps a fixed-size window of max_cache_len entries.
           Unlike PyTorch which allows truly unbounded growth, JAX requires
           a static buffer size for JIT compatibility.

    Attributes:
        num_heads: Number of attention heads.
        head_dim: Dimension per head for queries/keys.
        value_head_dim: Dimension per head for values.
        chunk_size: Size of attention chunks.
        max_cache_len: Maximum cache length. None/-1 = chunk_size.
        cache_unbounded: If True, disables chunk boundary resets (but buffer
            is still bounded by max_cache_len for JIT compatibility).
        attention_dropout: Dropout rate for attention weights.
        rotary: RotaryEmbedding module.
    """

    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    value_head_dim: int = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    max_cache_len: int = eqx.field(static=True)
    cache_unbounded: bool = eqx.field(static=True)
    attention_dropout: float = eqx.field(static=True)

    rotary: RotaryEmbedding

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        value_head_dim: int,
        chunk_size: int,
        rope_base: float = 10000.0,
        max_cache_len: int | None = None,
        cache_unbounded: bool = False,
        attention_dropout: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize ChunkedAttention.

        Args:
            num_heads: Number of attention heads.
            head_dim: Dimension per head for queries/keys.
            value_head_dim: Dimension per head for values.
            chunk_size: Size of attention chunks.
            rope_base: Base for rotary embeddings.
            max_cache_len: Maximum cache length. None/-1 = chunk_size.
                Values > chunk_size enable sliding window attention.
            cache_unbounded: If True, disables chunk boundary resets. Cache
                is still bounded by max_cache_len for JIT compatibility.
            attention_dropout: Dropout rate for attention weights.
            key: PRNG key for initialization.
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.value_head_dim = value_head_dim
        self.chunk_size = chunk_size
        # Handle None/-1 as "use chunk_size"
        if max_cache_len is None or max_cache_len < 0:
            self.max_cache_len = chunk_size
        else:
            self.max_cache_len = max_cache_len
        self.cache_unbounded = cache_unbounded
        self.attention_dropout = attention_dropout
        self.rotary = RotaryEmbedding(dim=head_dim, base=rope_base)

    def __call__(
        self,
        q: Float[Array, "batch seq heads head_dim"],
        k: Float[Array, "batch seq heads head_dim"],
        v: Float[Array, "batch seq heads value_dim"],
        cache: AttentionCache | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> tuple[
        Float[Array, "batch seq heads value_dim"],
        AttentionCache | None,
        Int[Array, ""] | None,
    ]:
        """Apply chunked attention with optional caching.

        Args:
            q: Query tensor (batch, seq, heads, head_dim).
            k: Key tensor (batch, seq, heads, head_dim).
            v: Value tensor (batch, seq, heads, value_dim).
            cache: Optional cached K/V from previous tokens.
            mask: Optional mask where True marks valid positions.
            return_cache: Whether to return updated cache.
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Tuple of (output, updated_cache, new_position).
        """
        B, L, H, _ = q.shape

        # Determine current position
        if cache is None:
            position = jnp.array(0, dtype=jnp.int32)
        else:
            position = cache.count

        # Training path: no cache, use multi-chunk attention
        if cache is None and not return_cache:
            out = attention_multi_chunk(
                q,
                k,
                v,
                chunk_size=self.chunk_size,
                start_index=position,
                rotary=self.rotary,
                mask=mask,
                dropout_rate=self.attention_dropout,
                deterministic=deterministic,
                key=key,
            )
            return out, None, position + L

        Dv = v.shape[-1]
        cache_size = self.max_cache_len

        # Determine cache reset behavior (matching PyTorch faithful_chunk_local)
        faithful_chunk_local = (not self.cache_unbounded) and (
            self.max_cache_len == self.chunk_size
        )

        # Streaming path: chunk-wise processing with fixed-size cache.
        # Full chunks use a single attention matmul; partial chunks fall back
        # to token-wise updates to avoid dynamic slice sizes.
        #
        # Cache behavior is controlled by cache_unbounded and max_cache_len:
        # - cache_unbounded=False, max_cache_len=chunk_size: "faithful_chunk_local" mode,
        #   resets KV at chunk boundaries (block-diagonal attention)
        # - cache_unbounded=False, max_cache_len>chunk_size: sliding window mode
        # - cache_unbounded=True: same as sliding window but no boundary resets
        #
        # Buffer model: circular ring buffer with masked validity.
        # Ordered cache views are materialized only when causal chunk attention
        # requires chronological K/V layout.
        #
        # Mask semantics:
        # - Cached prefix keys are always valid for the current chunk.
        # - Masked tokens in the current chunk are excluded from attention in that chunk,
        #   but their keys are still cached for future chunks.

        # Initialize or extract fixed-size cache buffers (ring order)
        if cache is None:
            cache_k = jnp.zeros((B, cache_size, H, self.head_dim), dtype=q.dtype)
            cache_v = jnp.zeros((B, cache_size, H, Dv), dtype=v.dtype)
        else:
            existing_len = cache.k.shape[1]
            if existing_len == cache_size:
                cache_k = cache.k
                cache_v = cache.v
            elif existing_len < cache_size:
                pad_len = cache_size - existing_len
                cache_k = jnp.pad(cache.k, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
                cache_v = jnp.pad(cache.v, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            else:
                # Keep the most recent entries
                cache_k = cache.k[:, -cache_size:]
                cache_v = cache.v[:, -cache_size:]

        has_input_mask = mask is not None
        cache_indices = jnp.arange(cache_size, dtype=jnp.int32)

        def current_valid_len(cur_pos: Int[Array, ""]) -> Int[Array, ""]:
            """Return the valid cache length at the current position."""
            if faithful_chunk_local:
                return jnp.minimum(cur_pos % self.chunk_size, cache_size)
            return jnp.minimum(cur_pos, cache_size)

        def cache_start(cur_pos: Int[Array, ""], valid_len: Int[Array, ""]) -> Int[Array, ""]:
            """Oldest cache index for the current window."""
            return jnp.mod(cur_pos - valid_len, cache_size)

        def ordered_cache(
            c_k: Float[Array, "batch cache_size heads head_dim"],
            c_v: Float[Array, "batch cache_size heads value_dim"],
            cur_pos: Int[Array, ""],
            valid_len: Int[Array, ""],
        ) -> tuple[
            Float[Array, "batch cache_size heads head_dim"],
            Float[Array, "batch cache_size heads value_dim"],
        ]:
            """Return cache K/V in chronological order for causal attention."""
            start = cache_start(cur_pos, valid_len)
            order = jnp.mod(start + cache_indices, cache_size)
            return jnp.take(c_k, order, axis=1), jnp.take(c_v, order, axis=1)

        def cache_mask_for(valid_len: Int[Array, ""]) -> Bool[Array, "batch cache_size"]:
            """Build a broadcasted cache validity mask."""
            mask_vec = cache_indices < valid_len
            return jnp.broadcast_to(mask_vec, (B, cache_size))

        def cache_mask_ring(
            cur_pos: Int[Array, ""], valid_len: Int[Array, ""]
        ) -> Bool[Array, "batch cache_size"]:
            """Cache validity mask for ring-ordered buffers."""
            start = cache_start(cur_pos, valid_len)
            age = jnp.mod(cache_indices - start, cache_size)
            mask_vec = age < valid_len
            return jnp.broadcast_to(mask_vec, (B, cache_size))

        def append_token(
            c_k: Float[Array, "batch cache_size heads head_dim"],
            c_v: Float[Array, "batch cache_size heads value_dim"],
            k_tok: Float[Array, "batch 1 heads head_dim"],
            v_tok: Float[Array, "batch 1 heads value_dim"],
            cur_pos: Int[Array, ""],
            valid_len: Int[Array, ""],
        ) -> tuple[
            Float[Array, "batch cache_size heads head_dim"],
            Float[Array, "batch cache_size heads value_dim"],
            Int[Array, ""],
            Int[Array, ""],
        ]:
            """Append a single token to the ring buffer."""
            write_pos = jnp.mod(cur_pos, cache_size).astype(jnp.int32)
            new_k = c_k.at[:, write_pos].set(k_tok[:, 0])
            new_v = c_v.at[:, write_pos].set(v_tok[:, 0])
            new_valid_len = jnp.minimum(valid_len + 1, cache_size)
            return new_k, new_v, write_pos, new_valid_len

        if L < self.chunk_size:
            # Small-L fast path: avoid padding to chunk_size and trace only token updates.
            out_buffer = jnp.zeros((B, L, H, Dv), dtype=v.dtype)

            def token_body(
                i: Int[Array, ""],
                state: tuple[
                    Float[Array, "batch seq heads value_dim"],
                    Float[Array, "batch cache_size heads head_dim"],
                    Float[Array, "batch cache_size heads value_dim"],
                    Int[Array, ""],
                    PRNGKeyArray | None,
                ],
            ) -> tuple[
                Float[Array, "batch seq heads value_dim"],
                Float[Array, "batch cache_size heads head_dim"],
                Float[Array, "batch cache_size heads value_dim"],
                Int[Array, ""],
                PRNGKeyArray | None,
            ]:
                """Process one token in the small-L streaming path."""
                out_step, c_k_step, c_v_step, pos_step, rng_step = state

                q_i = jax.lax.dynamic_slice(q, (0, i, 0, 0), (B, 1, H, self.head_dim))
                k_i = jax.lax.dynamic_slice(k, (0, i, 0, 0), (B, 1, H, self.head_dim))
                v_i = jax.lax.dynamic_slice(v, (0, i, 0, 0), (B, 1, H, Dv))

                q_rot, k_rot = self.rotary(q_i, k_i, pos_step)

                valid_len = current_valid_len(pos_step)
                c_k_step, c_v_step, write_pos, valid_len_new = append_token(
                    c_k_step, c_v_step, k_rot, v_i, pos_step, valid_len
                )
                cache_mask = cache_mask_ring(pos_step + 1, valid_len_new)
                if has_input_mask:
                    mask_i = jax.lax.dynamic_slice(mask, (0, i), (B, 1))[:, 0]
                    write_slot = cache_indices == write_pos
                    current_invalid = ~mask_i[:, None] & write_slot[None, :]
                    kv_mask = cache_mask & ~current_invalid
                else:
                    kv_mask = cache_mask

                if rng_step is not None:
                    rng_step, step_rng = jax.random.split(rng_step)
                else:
                    step_rng = None

                out_i = attention_single_chunk(
                    q_rot,
                    c_k_step,
                    c_v_step,
                    kv_mask=kv_mask,
                    causal=False,
                    dropout_rate=self.attention_dropout,
                    deterministic=deterministic,
                    key=step_rng,
                )

                out_step = jax.lax.dynamic_update_slice(out_step, out_i, (0, i, 0, 0))
                return out_step, c_k_step, c_v_step, pos_step + 1, rng_step

            init_state = (out_buffer, cache_k, cache_v, position, key)
            out_buffer, final_k, final_v, final_pos, _ = jax.lax.fori_loop(
                0, L, token_body, init_state
            )

            new_cache = AttentionCache(k=final_k, v=final_v, count=final_pos)
            if not return_cache:
                new_cache = None

            return out_buffer, new_cache, final_pos

        q_full = q
        k_full = k
        v_full = v
        mask_full = mask

        buffer_len = q_full.shape[1]
        out_buffer = jnp.zeros((B, buffer_len, H, Dv), dtype=v.dtype)
        full_chunk_mask = jnp.ones((B, self.chunk_size), dtype=jnp.bool_)

        if cache_size < self.chunk_size:

            def append_full_chunk(
                c_k: Float[Array, "batch cache_size heads head_dim"],
                c_v: Float[Array, "batch cache_size heads value_dim"],
                k_blk: Float[Array, "batch chunk_size heads head_dim"],
                v_blk: Float[Array, "batch chunk_size heads value_dim"],
                cur_pos: Int[Array, ""],
            ) -> tuple[
                Float[Array, "batch cache_size heads head_dim"],
                Float[Array, "batch cache_size heads value_dim"],
            ]:
                """Append a full chunk, keeping only the most recent cache_size tokens."""
                tail_offset = self.chunk_size - cache_size
                k_keep = k_blk[:, tail_offset:]
                v_keep = v_blk[:, tail_offset:]
                start_pos = cur_pos + tail_offset
                write_idx = jnp.mod(start_pos + cache_indices, cache_size)
                new_k = c_k.at[:, write_idx].set(k_keep)
                new_v = c_v.at[:, write_idx].set(v_keep)
                return new_k, new_v

        else:

            def append_full_chunk(
                c_k: Float[Array, "batch cache_size heads head_dim"],
                c_v: Float[Array, "batch cache_size heads value_dim"],
                k_blk: Float[Array, "batch chunk_size heads head_dim"],
                v_blk: Float[Array, "batch chunk_size heads value_dim"],
                cur_pos: Int[Array, ""],
            ) -> tuple[
                Float[Array, "batch cache_size heads head_dim"],
                Float[Array, "batch cache_size heads value_dim"],
            ]:
                """Append a full chunk to the ring buffer."""
                write_pos = jnp.mod(cur_pos, cache_size)
                offsets = cache_indices[: self.chunk_size]
                write_idx = jnp.mod(write_pos + offsets, cache_size)
                new_k = c_k.at[:, write_idx].set(k_blk)
                new_v = c_v.at[:, write_idx].set(v_blk)
                return new_k, new_v

        def process_full_chunk(
            offset: Int[Array, ""],
            out_buf: Float[Array, "batch seq heads value_dim"],
            c_k: Float[Array, "batch cache_size heads head_dim"],
            c_v: Float[Array, "batch cache_size heads value_dim"],
            cur_pos: Int[Array, ""],
            rng: PRNGKeyArray | None,
        ) -> tuple[
            Float[Array, "batch seq heads value_dim"],
            Float[Array, "batch cache_size heads head_dim"],
            Float[Array, "batch cache_size heads value_dim"],
            Int[Array, ""],
            PRNGKeyArray | None,
        ]:
            """Process a full chunk with cache and update the streaming state."""
            q_blk = jax.lax.dynamic_slice(
                q_full, (0, offset, 0, 0), (B, self.chunk_size, H, self.head_dim)
            )
            k_blk = jax.lax.dynamic_slice(
                k_full, (0, offset, 0, 0), (B, self.chunk_size, H, self.head_dim)
            )
            v_blk = jax.lax.dynamic_slice(v_full, (0, offset, 0, 0), (B, self.chunk_size, H, Dv))

            q_rot, k_rot = self.rotary(q_blk, k_blk, cur_pos)

            valid_len = current_valid_len(cur_pos)
            cache_mask = cache_mask_for(valid_len)
            mask_blk = None
            if has_input_mask:
                mask_blk = jax.lax.dynamic_slice(mask_full, (0, offset), (B, self.chunk_size))

            if mask_blk is not None:
                kv_mask = jnp.concatenate([cache_mask, mask_blk], axis=1)
            else:
                kv_mask = jnp.concatenate([cache_mask, full_chunk_mask], axis=1)

            if rng is not None:
                rng, step_rng = jax.random.split(rng)
            else:
                step_rng = None

            def attend_with_cache(_: None) -> Float[Array, "batch chunk_size heads value_dim"]:
                """Attend with cache concatenated to the current chunk."""
                ordered_k, ordered_v = ordered_cache(c_k, c_v, cur_pos, valid_len)
                k_cat = jnp.concatenate([ordered_k, k_rot], axis=1)
                v_cat = jnp.concatenate([ordered_v, v_blk], axis=1)
                return attention_single_chunk(
                    q_rot,
                    k_cat,
                    v_cat,
                    kv_mask=kv_mask,
                    causal=True,
                    dropout_rate=self.attention_dropout,
                    deterministic=deterministic,
                    key=step_rng,
                )

            def attend_no_cache(_: None) -> Float[Array, "batch chunk_size heads value_dim"]:
                """Attend within the current chunk only."""
                return attention_single_chunk(
                    q_rot,
                    k_rot,
                    v_blk,
                    kv_mask=mask_blk if mask_blk is not None else None,
                    causal=True,
                    dropout_rate=self.attention_dropout,
                    deterministic=deterministic,
                    key=step_rng,
                )

            if faithful_chunk_local:
                out_blk = jax.lax.cond(valid_len > 0, attend_with_cache, attend_no_cache, None)
            else:
                out_blk = attend_with_cache(None)

            out_buf = jax.lax.dynamic_update_slice(out_buf, out_blk, (0, offset, 0, 0))

            c_k, c_v = append_full_chunk(c_k, c_v, k_rot, v_blk, cur_pos)
            cur_pos = cur_pos + self.chunk_size

            return out_buf, c_k, c_v, cur_pos, rng

        def process_partial_chunk(
            offset: Int[Array, ""],
            chunk_len: Int[Array, ""],
            out_buf: Float[Array, "batch seq heads value_dim"],
            c_k: Float[Array, "batch cache_size heads head_dim"],
            c_v: Float[Array, "batch cache_size heads value_dim"],
            cur_pos: Int[Array, ""],
            rng: PRNGKeyArray | None,
        ) -> tuple[
            Float[Array, "batch seq heads value_dim"],
            Float[Array, "batch cache_size heads head_dim"],
            Float[Array, "batch cache_size heads value_dim"],
            Int[Array, ""],
            PRNGKeyArray | None,
        ]:
            """Process a partial chunk token-by-token."""

            def token_body(
                i: Int[Array, ""],
                state: tuple[
                    Float[Array, "batch seq heads value_dim"],
                    Float[Array, "batch cache_size heads head_dim"],
                    Float[Array, "batch cache_size heads value_dim"],
                    Int[Array, ""],
                    PRNGKeyArray | None,
                ],
            ) -> tuple[
                Float[Array, "batch seq heads value_dim"],
                Float[Array, "batch cache_size heads head_dim"],
                Float[Array, "batch cache_size heads value_dim"],
                Int[Array, ""],
                PRNGKeyArray | None,
            ]:
                """Process one token in a partial chunk."""
                out_step, c_k_step, c_v_step, pos_step, rng_step = state
                idx = offset + i

                q_i = jax.lax.dynamic_slice(q_full, (0, idx, 0, 0), (B, 1, H, self.head_dim))
                k_i = jax.lax.dynamic_slice(k_full, (0, idx, 0, 0), (B, 1, H, self.head_dim))
                v_i = jax.lax.dynamic_slice(v_full, (0, idx, 0, 0), (B, 1, H, Dv))

                q_rot, k_rot = self.rotary(q_i, k_i, pos_step)

                valid_len = current_valid_len(pos_step)
                c_k_step, c_v_step, write_pos, valid_len_new = append_token(
                    c_k_step, c_v_step, k_rot, v_i, pos_step, valid_len
                )
                cache_mask = cache_mask_ring(pos_step + 1, valid_len_new)
                if has_input_mask:
                    mask_i = jax.lax.dynamic_slice(mask_full, (0, idx), (B, 1))[:, 0]
                    write_slot = cache_indices == write_pos
                    current_invalid = ~mask_i[:, None] & write_slot[None, :]
                    kv_mask = cache_mask & ~current_invalid
                else:
                    kv_mask = cache_mask

                if rng_step is not None:
                    rng_step, step_rng = jax.random.split(rng_step)
                else:
                    step_rng = None

                out_i = attention_single_chunk(
                    q_rot,
                    c_k_step,
                    c_v_step,
                    kv_mask=kv_mask,
                    causal=False,
                    dropout_rate=self.attention_dropout,
                    deterministic=deterministic,
                    key=step_rng,
                )

                out_step = jax.lax.dynamic_update_slice(out_step, out_i, (0, idx, 0, 0))
                return out_step, c_k_step, c_v_step, pos_step + 1, rng_step

            init_state = (out_buf, c_k, c_v, cur_pos, rng)
            out_buf, c_k, c_v, cur_pos, rng = jax.lax.fori_loop(
                0, chunk_len, token_body, init_state
            )
            return out_buf, c_k, c_v, cur_pos, rng

        def loop_cond(
            state: tuple[Int[Array, ""], Array, Array, Array, Int[Array, ""], PRNGKeyArray | None],
        ) -> Bool[Array, ""]:
            """Continue streaming loop while offset is within sequence length."""
            offset = state[0]
            return offset < L

        def loop_body(
            state: tuple[
                Int[Array, ""],
                Float[Array, "batch seq heads value_dim"],
                Float[Array, "batch cache_size heads head_dim"],
                Float[Array, "batch cache_size heads value_dim"],
                Int[Array, ""],
                PRNGKeyArray | None,
            ],
        ) -> tuple[
            Int[Array, ""],
            Float[Array, "batch seq heads value_dim"],
            Float[Array, "batch cache_size heads head_dim"],
            Float[Array, "batch cache_size heads value_dim"],
            Int[Array, ""],
            PRNGKeyArray | None,
        ]:
            """Streaming loop body that dispatches full vs partial chunk processing."""
            offset, out_buf, c_k, c_v, cur_pos, rng = state
            remaining = L - offset
            if faithful_chunk_local:
                pos_in_chunk = cur_pos % self.chunk_size
                chunk_len = jnp.minimum(
                    remaining,
                    jnp.where(pos_in_chunk == 0, self.chunk_size, self.chunk_size - pos_in_chunk),
                )
            else:
                chunk_len = jnp.minimum(remaining, self.chunk_size)

            def full_branch(
                branch_state: tuple[
                    Int[Array, ""],
                    Float[Array, "batch seq heads value_dim"],
                    Float[Array, "batch cache_size heads head_dim"],
                    Float[Array, "batch cache_size heads value_dim"],
                    Int[Array, ""],
                    PRNGKeyArray | None,
                ],
            ) -> tuple[
                Int[Array, ""],
                Float[Array, "batch seq heads value_dim"],
                Float[Array, "batch cache_size heads head_dim"],
                Float[Array, "batch cache_size heads value_dim"],
                Int[Array, ""],
                PRNGKeyArray | None,
            ]:
                """Branch for processing a full chunk."""
                offset_b, out_b, k_b, v_b, pos_b, rng_b = branch_state
                out_b, k_b, v_b, pos_b, rng_b = process_full_chunk(
                    offset_b, out_b, k_b, v_b, pos_b, rng_b
                )
                offset_b = offset_b + self.chunk_size
                return offset_b, out_b, k_b, v_b, pos_b, rng_b

            def partial_branch(
                branch_state: tuple[
                    Int[Array, ""],
                    Float[Array, "batch seq heads value_dim"],
                    Float[Array, "batch cache_size heads head_dim"],
                    Float[Array, "batch cache_size heads value_dim"],
                    Int[Array, ""],
                    PRNGKeyArray | None,
                ],
            ) -> tuple[
                Int[Array, ""],
                Float[Array, "batch seq heads value_dim"],
                Float[Array, "batch cache_size heads head_dim"],
                Float[Array, "batch cache_size heads value_dim"],
                Int[Array, ""],
                PRNGKeyArray | None,
            ]:
                """Branch for processing a partial chunk."""
                offset_b, out_b, k_b, v_b, pos_b, rng_b = branch_state
                out_b, k_b, v_b, pos_b, rng_b = process_partial_chunk(
                    offset_b, chunk_len, out_b, k_b, v_b, pos_b, rng_b
                )
                offset_b = offset_b + chunk_len
                return offset_b, out_b, k_b, v_b, pos_b, rng_b

            do_full = chunk_len == self.chunk_size
            return jax.lax.cond(do_full, full_branch, partial_branch, state)

        init_offset = jnp.array(0, dtype=jnp.int32)
        init_state = (init_offset, out_buffer, cache_k, cache_v, position, key)
        _, out_buffer, final_k, final_v, final_pos, _ = jax.lax.while_loop(
            loop_cond, loop_body, init_state
        )

        if buffer_len > L:
            out_buffer = out_buffer[:, :L, :, :]

        new_cache = AttentionCache(k=final_k, v=final_v, count=final_pos)
        if not return_cache:
            new_cache = None

        return out_buffer, new_cache, final_pos


# -----------------------------------------------------------------------------
# MegalodonAttention Block
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
    z_dim: int = eqx.field(static=True)
    value_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)
    head_dim: int = eqx.field(static=True)
    value_head_dim: int = eqx.field(static=True)
    norm_eps: float = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    hidden_dropout: float = eqx.field(static=True)

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
        max_cache_len: int | None = None,
        cache_unbounded: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        *,
        key: PRNGKeyArray,
    ):
        """Initialize MegalodonAttention.

        Args:
            model_dim: Model hidden dimension D.
            z_dim: Shared Q/K dimension.
            value_dim: Value dimension.
            num_heads: Number of attention heads.
            cema_ndim: Number of EMA orders for ComplexEMA.
            chunk_size: Attention chunk size.
            norm_num_groups: Number of groups for TimestepNorm.
            norm_eps: Epsilon for normalization.
            norm_affine: Whether normalization layers include affine parameters.
            rope_base: Base for rotary embeddings.
            max_cache_len: Max KV cache length. None/-1 = chunk_size.
            cache_unbounded: If True, never clamp cache length.
            dropout: Output dropout rate.
            attention_dropout: Attention weight dropout rate.
            hidden_dropout: Hidden layer dropout rate.
            key: PRNG key for initialization.
        """
        self.model_dim = model_dim
        self.z_dim = z_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = z_dim // num_heads
        self.value_head_dim = value_dim // num_heads
        self.norm_eps = norm_eps
        self.dropout = dropout
        self.hidden_dropout = hidden_dropout

        # Split keys
        keys = jax.random.split(key, 9)

        # Sub-modules
        self.timenorm = TimestepNorm(
            num_features=model_dim, num_groups=norm_num_groups, eps=norm_eps, affine=norm_affine
        )
        self.cema = ComplexEMA(embed_dim=model_dim, ndim=cema_ndim, key=keys[0])
        self.rmsnorm = RMSNorm(dim=model_dim, eps=norm_eps, affine=norm_affine)
        self.inner = ChunkedAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            value_head_dim=self.value_head_dim,
            chunk_size=chunk_size,
            rope_base=rope_base,
            max_cache_len=max_cache_len,
            cache_unbounded=cache_unbounded,
            attention_dropout=attention_dropout,
            key=keys[1],
        )

        # Projections
        self.wz = eqx.nn.Linear(model_dim, z_dim, key=keys[2])
        self.wv = eqx.nn.Linear(model_dim, value_dim, key=keys[3])
        self.wr = eqx.nn.Linear(model_dim, value_dim, key=keys[4])
        self.wh1 = eqx.nn.Linear(model_dim, model_dim, key=keys[5])
        self.wh2 = eqx.nn.Linear(value_dim, model_dim, key=keys[6])

        # Per-head affine parameters (gamma+1 parameterization, init zeros)
        self.gamma = jnp.zeros((2, z_dim))
        self.beta = jnp.zeros((2, z_dim))

    def __call__(
        self,
        x: Float[Array, "batch seq dim"],
        cache: LayerCache | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
        return_cache: bool = False,
        deterministic: bool = True,
        key: PRNGKeyArray | None = None,
    ) -> tuple[Float[Array, "batch seq dim"], LayerCache | None]:
        """Forward pass through the attention block.

        Args:
            x: Input tensor of shape (batch, seq, dim).
            cache: Optional layer cache from previous tokens.
            mask: Optional mask where True marks valid positions.
            return_cache: Whether to return updated cache.
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Tuple of (output, updated_cache).
        """
        B, L, D = x.shape
        H = self.num_heads
        Dh = self.head_dim
        Dv = self.value_head_dim

        # Split keys for dropout (4 keys: attention, mx_dropout, gated_dropout, output_dropout)
        if key is not None:
            k1, k2, k3, k4 = jax.random.split(key, 4)
        else:
            k1 = k2 = k3 = k4 = None

        # Extract cache components
        norm_state = cache.norm if cache is not None else None
        ema_state = cache.ema.h if cache is not None and cache.ema is not None else None
        attn_cache = cache.attn if cache is not None else None

        # TimestepNorm
        x_tn, new_norm_state = self.timenorm(x, state=norm_state, mask=mask)

        # CEMA: (B, L, D) -> (B, D, L) -> CEMA -> (B, D, L) -> (B, L, D)
        # Pass mask to prevent EMA state contamination from padded positions
        need_ema_state = return_cache or ema_state is not None
        y_cema, h_last = self.cema(
            x_tn.transpose(0, 2, 1),  # (B, D, L)
            h_init=ema_state,
            return_state=need_ema_state,
            mask=mask,  # (B, L) - zeros masked positions to prevent state contamination
        )
        y_cema = y_cema.transpose(0, 2, 1)  # (B, L, D)

        # RMSNorm on CEMA output, then hidden_dropout (matching PyTorch reference line 1370)
        mx = self.rmsnorm(y_cema)
        if not deterministic and self.hidden_dropout > 0.0 and k2 is not None:
            keep = jax.random.bernoulli(k2, 1.0 - self.hidden_dropout, mx.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - self.hidden_dropout), dtype=mx.dtype)
            mx = jnp.where(keep, mx * inv_keep, jnp.zeros((), dtype=mx.dtype))

        # Shared Z projection for Q/K
        z = _linear_3d(self.wz, mx)  # (B, L, z_dim)
        z = z.reshape(B, L, H, Dh)

        # Per-head RMS normalization (in fp32 for stability)
        z_f32 = z.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(z_f32**2, axis=-1, keepdims=True) + self.norm_eps)
        z_normed = (z_f32 / rms).astype(z.dtype)

        # Affine transform to Q and K
        # gamma, beta: (2, z_dim) -> (2, H, Dh)
        # Cast to z.dtype to preserve bf16 in mixed-precision
        gamma_heads = self.gamma.reshape(2, H, Dh).astype(z.dtype)
        beta_heads = self.beta.reshape(2, H, Dh).astype(z.dtype)
        scale = (gamma_heads + 1.0) / math.sqrt(Dh)

        # Broadcast and apply: z_normed is (B, L, H, Dh)
        q = z_normed * scale[0] + beta_heads[0]
        k = z_normed * scale[1] + beta_heads[1]

        # Value projection with SiLU
        v = jax.nn.silu(_linear_3d(self.wv, x_tn))  # (B, L, value_dim)
        v = v.reshape(B, L, H, Dv)

        # Gate projection with SiLU
        r = jax.nn.silu(_linear_3d(self.wr, mx))  # (B, L, value_dim)

        # Inner attention
        out, new_attn_cache, new_position = self.inner(
            q,
            k,
            v,
            cache=attn_cache,
            mask=mask,
            return_cache=return_cache,
            deterministic=deterministic,
            key=k1,
        )

        # Reshape attention output: (B, L, H, Dv) -> (B, L, value_dim)
        out = out.reshape(B, L, self.value_dim)

        # Apply gate
        gated = out * r

        # Hidden dropout on gated attention output (matching PyTorch reference line 1418)
        if not deterministic and self.hidden_dropout > 0.0 and k3 is not None:
            keep = jax.random.bernoulli(k3, 1.0 - self.hidden_dropout, gated.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - self.hidden_dropout), dtype=gated.dtype)
            gated = jnp.where(keep, gated * inv_keep, jnp.zeros((), dtype=gated.dtype))

        # Output projections
        h = _linear_3d(self.wh1, mx) + _linear_3d(self.wh2, gated)

        # Output dropout
        if not deterministic and self.dropout > 0.0 and k4 is not None:
            keep = jax.random.bernoulli(k4, 1.0 - self.dropout, h.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - self.dropout), dtype=h.dtype)
            h = jnp.where(keep, h * inv_keep, jnp.zeros((), dtype=h.dtype))

        # Residual
        y = h + x

        # Build output cache
        if return_cache:
            new_cache = LayerCache(
                attn=new_attn_cache,
                norm=new_norm_state,
                ema=EMAState(h=h_last) if h_last is not None else None,
                position=new_position
                if new_position is not None
                else jnp.array(0, dtype=jnp.int32),
            )
        else:
            new_cache = None

        return y, new_cache

    @classmethod
    def with_init(
        cls,
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
        max_cache_len: int | None = None,
        cache_unbounded: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        init_mode: InitMode = "gaussian",
        *,
        key: PRNGKeyArray,
    ) -> "MegalodonAttention":
        """Create MegalodonAttention with custom weight initialization.

        This is the recommended way to construct MegalodonAttention when using
        custom initialization modes. Creates the module with default init,
        then reinitializes all Linear weights according to init_mode.

        Args:
            Same as __init__, plus:
            init_mode: Initialization mode for Linear weights. One of:
                - "gaussian": Truncated normal with stddev=1/sqrt(dim) when dim is set,
                  or stddev=1.0 when dim is None (PyTorch parity)
                - "xavier": Glorot uniform (Equinox default)
                - "he": He normal (for ReLU networks)
                - "bert": Normal with stddev=0.02
                - "none": Skip reinitialization (use Equinox default)

        Returns:
            MegalodonAttention instance with reinitialized weights.
        """
        key1, key2 = jax.random.split(key)
        instance = cls(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            rope_base=rope_base,
            max_cache_len=max_cache_len,
            cache_unbounded=cache_unbounded,
            dropout=dropout,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            key=key1,
        )
        if init_mode != "none":
            # Don't pass dim - matches PyTorch behavior (std=1.0 for gaussian)
            instance = reinit_linear_weights(instance, init_mode, key2)
        return instance


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

    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    fc3: eqx.nn.Linear | None  # Only for SwiGLU

    model_dim: int = eqx.field(static=True)
    ffn_hidden_dim: int = eqx.field(static=True)
    swiglu: bool = eqx.field(static=True)
    hidden_dropout: float = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    alpha: float | None = eqx.field(static=True)  # Residual rescale factor

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
        *,
        key: PRNGKeyArray,
    ):
        """Initialize NormalizedFFN.

        Args:
            model_dim: Model hidden dimension.
            ffn_hidden_dim: FFN intermediate dimension.
            norm_eps: Epsilon for layer normalization.
            norm_affine: Whether normalization includes affine parameters.
            swiglu: Whether to use SwiGLU activation.
            rescale: Whether to apply residual rescaling (rescale_nffn).
            layer_id: Layer index for computing rescale factor (0-indexed).
            hidden_dropout: Dropout after hidden layer.
            dropout: Dropout after output projection.
            key: PRNG key for initialization.
        """
        self.model_dim = model_dim
        self.ffn_hidden_dim = ffn_hidden_dim
        self.swiglu = swiglu
        self.hidden_dropout = hidden_dropout
        self.dropout = dropout
        # Compute rescale alpha: 0.1 * (0.5 ** layer_id)
        self.alpha = (0.1 * (0.5**layer_id)) if rescale else None

        keys = jax.random.split(key, 3)

        self.norm = eqx.nn.LayerNorm(
            shape=model_dim, eps=norm_eps, use_weight=norm_affine, use_bias=norm_affine
        )
        self.fc1 = eqx.nn.Linear(model_dim, ffn_hidden_dim, key=keys[0])
        self.fc2 = eqx.nn.Linear(ffn_hidden_dim, model_dim, key=keys[1])

        if swiglu:
            self.fc3 = eqx.nn.Linear(model_dim, ffn_hidden_dim, key=keys[2])
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

        Args:
            x: Input tensor of shape (batch, seq, dim).
            residual_base: Optional tensor to use as residual base (two-hop).
            deterministic: If True, skip dropout.
            key: PRNG key for dropout.

        Returns:
            Output tensor of shape (batch, seq, dim).
        """
        residual = x if residual_base is None else residual_base

        # Layer norm
        # eqx.nn.LayerNorm expects per-example inputs, so vmap over batch/seq.
        h = jax.vmap(jax.vmap(self.norm))(x)

        # Hidden layer with activation
        if self.swiglu:
            h = jax.nn.silu(_linear_3d(self.fc1, h)) * _linear_3d(self.fc3, h)
        else:
            h = jax.nn.silu(_linear_3d(self.fc1, h))

        # Hidden dropout
        if not deterministic and self.hidden_dropout > 0.0:
            if key is not None:
                k1, k2 = jax.random.split(key)
            else:
                k1 = k2 = None
            if k1 is not None:
                keep = jax.random.bernoulli(k1, 1.0 - self.hidden_dropout, h.shape)
                inv_keep = jnp.asarray(1.0 / (1.0 - self.hidden_dropout), dtype=h.dtype)
                h = jnp.where(keep, h * inv_keep, jnp.zeros((), dtype=h.dtype))
        else:
            k2 = key

        # Output projection
        out = _linear_3d(self.fc2, h)

        # Output dropout
        if not deterministic and self.dropout > 0.0 and k2 is not None:
            keep = jax.random.bernoulli(k2, 1.0 - self.dropout, out.shape)
            inv_keep = jnp.asarray(1.0 / (1.0 - self.dropout), dtype=out.dtype)
            out = jnp.where(keep, out * inv_keep, jnp.zeros((), dtype=out.dtype))

        # Apply residual rescaling if enabled (cast to preserve bf16)
        if self.alpha is not None:
            out = out * jnp.asarray(self.alpha, dtype=out.dtype)

        return residual + out

    @classmethod
    def with_init(
        cls,
        model_dim: int,
        ffn_hidden_dim: int,
        norm_eps: float = 1e-5,
        norm_affine: bool = True,
        swiglu: bool = False,
        rescale: bool = False,
        layer_id: int = 0,
        hidden_dropout: float = 0.0,
        dropout: float = 0.0,
        init_mode: InitMode = "gaussian",
        *,
        key: PRNGKeyArray,
    ) -> "NormalizedFFN":
        """Create NormalizedFFN with custom weight initialization.

        This is the recommended way to construct NormalizedFFN when using
        custom initialization modes. Creates the module with default init,
        then reinitializes all Linear weights according to init_mode.

        Args:
            Same as __init__, plus:
            init_mode: Initialization mode for Linear weights. One of:
                - "gaussian": Truncated normal with stddev=1/sqrt(dim) when dim is set,
                  or stddev=1.0 when dim is None (PyTorch parity)
                - "xavier": Glorot uniform (Equinox default)
                - "he": He normal (for ReLU networks)
                - "bert": Normal with stddev=0.02
                - "none": Skip reinitialization (use Equinox default)

        Returns:
            NormalizedFFN instance with reinitialized weights.
        """
        key1, key2 = jax.random.split(key)
        instance = cls(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_hidden_dim,
            norm_eps=norm_eps,
            norm_affine=norm_affine,
            swiglu=swiglu,
            rescale=rescale,
            layer_id=layer_id,
            hidden_dropout=hidden_dropout,
            dropout=dropout,
            key=key1,
        )
        if init_mode != "none":
            # Don't pass dim - matches PyTorch behavior (std=1.0 for gaussian)
            instance = reinit_linear_weights(instance, init_mode, key2)
        return instance
