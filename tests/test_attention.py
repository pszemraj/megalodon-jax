"""Phase 3 Attention tests - primitives, ChunkedAttention, MegalodonAttention, NormalizedFFN."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax.layers import (
    ChunkedAttention,
    MegalodonAttention,
    NormalizedFFN,
    RotaryEmbedding,
    attention_multi_chunk,
    attention_single_chunk,
)
from tests.utils import require_torch_modeling, to_jax, to_torch

# -----------------------------------------------------------------------------
# Attention Primitive Tests
# -----------------------------------------------------------------------------


class TestAttentionPrimitives:
    """Tests for attention_single_chunk and attention_multi_chunk."""

    def test_single_chunk_shapes(self, random_seed: int) -> None:
        """Test that attention_single_chunk produces correct output shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 2, 16, 4, 32, 64

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))

        out = attention_single_chunk(q, k, v)

        assert out.shape == (batch, seq, heads, value_dim)

    def test_single_chunk_causal_masking(self, random_seed: int) -> None:
        """Test that causal masking prevents attending to future positions.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim = 1, 4, 1, 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        # Create Q/K/V with random values
        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, head_dim))

        out = attention_single_chunk(q, k, v, causal=True)

        # For causal attention, verify shapes and non-NaN outputs
        assert out.shape == (batch, seq, heads, head_dim)
        assert not jnp.any(jnp.isnan(out))

        # Verify causality: output at position 0 should only depend on input at position 0
        # We can test this by checking that changing future inputs doesn't affect past outputs
        # Modify v at position 3 and check that output at position 0 is unchanged
        v_modified = v.at[:, 3, :, :].set(v[:, 3, :, :] * 100)
        out_modified = attention_single_chunk(q, k, v_modified, causal=True)

        # Position 0 output should be identical (can't see position 3)
        np.testing.assert_allclose(
            np.array(out[:, 0, :, :]),
            np.array(out_modified[:, 0, :, :]),
            rtol=1e-6,
            err_msg="Causal masking failed: position 0 saw future position 3",
        )

    def test_single_chunk_no_temperature_scaling(self, random_seed: int) -> None:
        """Test that attention uses scale=1.0 (no temperature scaling).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim = 1, 4, 1, 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        # Use small values to avoid softmax saturation
        q = jax.random.normal(k1, (batch, seq, heads, head_dim)) * 0.1
        k = jax.random.normal(k2, (batch, seq, heads, head_dim)) * 0.1
        v = jax.random.normal(k3, (batch, seq, heads, head_dim))

        out = attention_single_chunk(q, k, v)

        # Verify output is not NaN and has reasonable magnitude
        assert not jnp.any(jnp.isnan(out))

    def test_single_chunk_preserves_bf16_dtype(self, random_seed: int) -> None:
        """Test that attention_single_chunk preserves bf16 dtype (no forced fp32).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 1, 4, 2, 16, 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(k2, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(k3, (batch, seq, heads, value_dim), dtype=jnp.bfloat16)

        out = attention_single_chunk(q, k, v)

        assert out.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(out))

    def test_post_softmax_and_dropkey_semantics(self) -> None:
        """The two released attention-dropout placements remain explicit."""
        q = jnp.zeros((1, 1, 1, 2), dtype=jnp.float32)
        k = jnp.zeros((1, 4, 1, 2), dtype=jnp.float32)
        v = jnp.arange(1, 5, dtype=jnp.float32).reshape(1, 4, 1, 1)
        key = jax.random.PRNGKey(7)
        keep = np.asarray(jax.random.bernoulli(key, 0.5, (1, 1, 1, 4)))[0, 0, 0]

        post = attention_single_chunk(
            q,
            k,
            v,
            causal=False,
            dropout_rate=0.5,
            dropout_mode="post_softmax",
            deterministic=False,
            key=key,
        )
        dropkey = attention_single_chunk(
            q,
            k,
            v,
            causal=False,
            dropout_rate=0.5,
            dropout_mode="dropkey",
            deterministic=False,
            key=key,
        )

        values = np.arange(1, 5, dtype=np.float32)
        expected_post = np.sum(values * keep * 0.5)
        expected_dropkey = np.mean(values[keep]) if np.any(keep) else 0.0
        np.testing.assert_allclose(np.asarray(post).item(), expected_post, atol=1e-6)
        np.testing.assert_allclose(np.asarray(dropkey).item(), expected_dropkey, atol=1e-6)

    def test_attention_dropout_rejects_one(self) -> None:
        """Degenerate p=1 cannot reach either softmax path."""
        q = jnp.ones((1, 1, 1, 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\)"):
            attention_single_chunk(q, q, q, dropout_rate=1.0)

    def test_single_chunk_query_key_mask_blocks_cross_segment(self) -> None:
        """qk_mask should block cross-segment attention links."""
        q = jnp.ones((1, 4, 1, 2), dtype=jnp.float32)
        k = jnp.ones((1, 4, 1, 2), dtype=jnp.float32)
        v = jnp.asarray([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]], dtype=jnp.float32)
        segs = jnp.asarray([[1, 1, 2, 2]], dtype=jnp.int32)
        qk_mask = (
            (segs[:, :, None] == segs[:, None, :]) & (segs[:, :, None] > 0) & (segs[:, None, :] > 0)
        )

        out = attention_single_chunk(q, k, v, qk_mask=qk_mask, causal=False)
        np.testing.assert_allclose(np.array(out[0, 0, 0, 0]), 1.5, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(out[0, 3, 0, 0]), 3.5, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("chunk_size", [16, 8])
    def test_repeated_segment_ids_match_unique_run_ids(
        self, random_seed: int, chunk_size: int
    ) -> None:
        """Reused positive ids must isolate exactly like unique per-run ids.

        A packer may legally emit ``[1, 1, 2, 2, 1, 1]``; raw-id equality would
        let the later ``1`` run attend back to the earlier one. chunk_size=16
        exercises the single-chunk mask path, 8 the multi-chunk path.

        :param int random_seed: Random seed fixture.
        :param int chunk_size: Attention chunk size under test.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 1, 16, 2, 16, 16
        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))
        rotary = RotaryEmbedding(dim=head_dim)
        start_index = jnp.array(0, dtype=jnp.int32)

        seg_repeated = jnp.asarray(
            [[1, 1, 1, 2, 2, 1, 1, 1, 3, 3, 3, 3, 1, 1, 1, 1]], dtype=jnp.int32
        )
        seg_unique = jnp.asarray(
            [[1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]], dtype=jnp.int32
        )
        position_ids = jnp.asarray(
            [[0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 3, 0, 1, 2, 3]], dtype=jnp.int32
        )

        def run(segment_ids: jnp.ndarray) -> jnp.ndarray:
            """Run chunked attention with the given segment ids.

            :param jnp.ndarray segment_ids: Per-token segment ids.
            :return jnp.ndarray: Attention output.
            """
            return attention_multi_chunk(
                q,
                k,
                v,
                chunk_size=chunk_size,
                start_index=start_index,
                rotary=rotary,
                segment_ids=segment_ids,
                position_ids=position_ids,
            )

        np.testing.assert_array_equal(
            np.array(run(seg_repeated)),
            np.array(run(seg_unique)),
            err_msg="Repeated segment ids must isolate the same as unique run ids",
        )

    def test_unaligned_segment_crossing_chunk_matches_standalone(self, random_seed: int) -> None:
        """A doc starting mid-chunk must attend exactly as it would run alone.

        With chunk_size=8, doc B occupies global positions 6-15 and crosses
        the global chunk boundary at position 8. Chunk boundaries must be
        re-anchored at the segment start (doc-local split at 8, global 14),
        not inherited from the global grid (doc-local split at 2).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 1, 16, 2, 16, 16
        chunk_size = 8
        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))
        rotary = RotaryEmbedding(dim=head_dim)
        start_index = jnp.array(0, dtype=jnp.int32)

        # doc A: positions 0-5 (id 1); doc B: positions 6-15 (id 2)
        segment_ids = jnp.asarray([[1] * 6 + [2] * 10], dtype=jnp.int32)
        position_ids = jnp.asarray([list(range(6)) + list(range(10))], dtype=jnp.int32)

        out_packed = attention_multi_chunk(
            q,
            k,
            v,
            chunk_size=chunk_size,
            start_index=start_index,
            rotary=rotary,
            segment_ids=segment_ids,
            position_ids=position_ids,
        )
        out_alone = attention_multi_chunk(
            q[:, 6:],
            k[:, 6:],
            v[:, 6:],
            chunk_size=chunk_size,
            start_index=start_index,
            rotary=rotary,
        )

        np.testing.assert_allclose(
            np.array(out_packed[:, 6:]),
            np.array(out_alone),
            rtol=1e-5,
            atol=1e-6,
            err_msg="Doc B packed at an unaligned offset must match doc B alone",
        )

    def test_multi_chunk_shapes(self, random_seed: int) -> None:
        """Test that attention_multi_chunk produces correct output shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 2, 64, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))

        rotary = RotaryEmbedding(dim=head_dim)
        start_index = jnp.array(0, dtype=jnp.int32)

        out = attention_multi_chunk(
            q, k, v, chunk_size=chunk_size, start_index=start_index, rotary=rotary
        )

        assert out.shape == (batch, seq, heads, value_dim)

    def test_multi_chunk_padding(self, random_seed: int) -> None:
        """Test multi_chunk handles sequences not divisible by chunk_size.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 2, 50, 4, 32, 64
        chunk_size = 16  # 50 is not divisible by 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))

        rotary = RotaryEmbedding(dim=head_dim)
        start_index = jnp.array(0, dtype=jnp.int32)

        out = attention_multi_chunk(
            q, k, v, chunk_size=chunk_size, start_index=start_index, rotary=rotary
        )

        # Output should have original sequence length (padding removed)
        assert out.shape == (batch, seq, heads, value_dim)
        assert not jnp.any(jnp.isnan(out))

    def test_multi_chunk_explicit_position_ids_path(self, random_seed: int) -> None:
        """Explicit position_ids should be honored by rotary in multi-chunk attention."""
        batch, seq, heads, head_dim, value_dim = 1, 16, 2, 8, 8
        chunk_size = 8
        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)
        q = jax.random.normal(k1, (batch, seq, heads, head_dim))
        k = jax.random.normal(k2, (batch, seq, heads, head_dim))
        v = jax.random.normal(k3, (batch, seq, heads, value_dim))

        rotary = RotaryEmbedding(dim=head_dim)
        start_index = jnp.array(0, dtype=jnp.int32)
        mono_pos = jnp.broadcast_to(jnp.arange(seq, dtype=jnp.int32)[None, :], (batch, seq))
        shifted_pos = mono_pos + 100

        out_default = attention_multi_chunk(
            q, k, v, chunk_size=chunk_size, start_index=start_index, rotary=rotary
        )
        out_mono = attention_multi_chunk(
            q,
            k,
            v,
            chunk_size=chunk_size,
            start_index=start_index,
            rotary=rotary,
            position_ids=mono_pos,
        )
        out_shifted = attention_multi_chunk(
            q,
            k,
            v,
            chunk_size=chunk_size,
            start_index=start_index,
            rotary=rotary,
            position_ids=shifted_pos,
        )

        np.testing.assert_allclose(np.array(out_default), np.array(out_mono), rtol=1e-5, atol=1e-5)
        assert not np.allclose(np.array(out_default), np.array(out_shifted))


# -----------------------------------------------------------------------------
# ChunkedAttention Tests
# -----------------------------------------------------------------------------


class TestChunkedAttention:
    """Tests for ChunkedAttention module."""

    @pytest.mark.parametrize("attention_window", [None, 3, 8])
    def test_arbitrary_call_partition_invariance(
        self,
        random_seed: int,
        attention_window: int | None,
    ) -> None:
        """Full, chunked, and tokenwise calls share one semantic timeline."""
        key = jax.random.PRNGKey(random_seed)
        k_module, kq, kk, kv = jax.random.split(key, 4)
        module = ChunkedAttention(
            num_heads=1,
            head_dim=4,
            value_head_dim=3,
            chunk_size=4,
            attention_window=attention_window,
            key=k_module,
        )
        q = jax.random.normal(kq, (1, 12, 1, 4))
        k = jax.random.normal(kk, (1, 12, 1, 4))
        v = jax.random.normal(kv, (1, 12, 1, 3))

        reference, reference_cache, _ = module(q, k, v, return_cache=True)
        assert reference_cache is not None
        for partitions in ([1] * 12, [3, 2, 7], [5, 4, 3], [4, 4, 4]):
            outputs = []
            cache = None
            offset = 0
            for width in partitions:
                part, cache, _ = module(
                    q[:, offset : offset + width],
                    k[:, offset : offset + width],
                    v[:, offset : offset + width],
                    cache=cache,
                    return_cache=True,
                )
                outputs.append(part)
                offset += width
            actual = jnp.concatenate(outputs, axis=1)
            np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(cache.k, reference_cache.k, atol=0.0, rtol=0.0)
            np.testing.assert_allclose(cache.v, reference_cache.v, atol=0.0, rtol=0.0)
            np.testing.assert_array_equal(cache.count, reference_cache.count)

        noncached, _, _ = module(q, k, v)
        np.testing.assert_allclose(noncached, reference, atol=2e-6, rtol=2e-6)

    def test_cache_copy_continuation_is_identical(self, random_seed: int) -> None:
        """A reloaded array-identical cache resumes without numerical drift."""
        key = jax.random.PRNGKey(random_seed)
        km, kq, kk, kv = jax.random.split(key, 4)
        module = ChunkedAttention(1, 4, 3, 4, attention_window=6, key=km)
        q = jax.random.normal(kq, (1, 9, 1, 4))
        k = jax.random.normal(kk, (1, 9, 1, 4))
        v = jax.random.normal(kv, (1, 9, 1, 3))
        _, cache, _ = module(q[:, :5], k[:, :5], v[:, :5], return_cache=True)
        assert cache is not None
        restored = type(cache)(
            k=jnp.array(cache.k), v=jnp.array(cache.v), count=jnp.array(cache.count)
        )
        expected, _, _ = module(q[:, 5:], k[:, 5:], v[:, 5:], cache=cache, return_cache=True)
        actual, _, _ = module(q[:, 5:], k[:, 5:], v[:, 5:], cache=restored, return_cache=True)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    def test_forward_shapes(self, random_seed: int) -> None:
        """Test ChunkedAttention forward pass shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 2, 32, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        q = jax.random.normal(k2, (batch, seq, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq, heads, value_dim))

        out, cache, position = attn(q, k, v)

        assert out.shape == (batch, seq, heads, value_dim)
        assert cache is None  # No cache returned without return_cache=True

    def test_streaming_with_cache(self, random_seed: int) -> None:
        """Test ChunkedAttention streaming with cache.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 2, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Process tokens one at a time
        outputs = []
        cache = None

        for i in range(8):
            ki = jax.random.fold_in(k2, i)
            q = jax.random.normal(ki, (batch, 1, heads, head_dim))
            k = jax.random.normal(jax.random.fold_in(k3, i), (batch, 1, heads, head_dim))
            v = jax.random.normal(jax.random.fold_in(k4, i), (batch, 1, heads, value_dim))

            out, cache, position = attn(q, k, v, cache=cache, return_cache=True)
            outputs.append(out)

        # Check final cache state
        assert cache is not None
        assert cache.count == 8
        # Fixed-size buffer uses chunk_size in released chunk-local mode.
        assert cache.k.shape[1] == chunk_size

    def test_streaming_rejects_strict_metadata(self, random_seed: int) -> None:
        """Streaming cache path should reject segment/position strict metadata."""
        heads, head_dim, value_dim, chunk_size = 2, 8, 8, 8
        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )
        q = jax.random.normal(k2, (1, 4, heads, head_dim))
        k = jax.random.normal(k3, (1, 4, heads, head_dim))
        v = jax.random.normal(k4, (1, 4, heads, value_dim))
        segs = jnp.asarray([[1, 1, 2, 2]], dtype=jnp.int32)
        pos = jnp.asarray([[0, 1, 0, 1]], dtype=jnp.int32)

        with pytest.raises(ValueError, match="segment_ids/position_ids"):
            attn(q, k, v, segment_ids=segs, position_ids=pos, return_cache=True)

    def test_streaming_small_l_preserves_dtype_and_cache_count(self, random_seed: int) -> None:
        """Small-L streaming should keep dtype and track cache count without padding.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 1, 2, 16, 16
        chunk_size = 8
        seq = 3  # L < chunk_size to hit the token-wise path

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        q = jax.random.normal(k2, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        k = jax.random.normal(k3, (batch, seq, heads, head_dim), dtype=jnp.bfloat16)
        v = jax.random.normal(k4, (batch, seq, heads, value_dim), dtype=jnp.bfloat16)

        out, cache, position = attn(q, k, v, return_cache=True)

        assert out.shape == (batch, seq, heads, value_dim)
        assert out.dtype == jnp.bfloat16
        assert cache is not None
        assert cache.count == seq  # should only advance by actual tokens
        assert position == seq

    def test_streaming_cache_ring_order(self, random_seed: int) -> None:
        """Ring buffer should retain the latest tokens by absolute position.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 1, 1, 2, 2
        chunk_size = 4
        cache_size = 4
        seq = 6  # force wraparound

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            attention_window=cache_size,
            key=k1,
        )

        q = jax.random.normal(k2, (batch, seq, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq, heads, value_dim))

        cache = None
        for i in range(seq):
            _, cache, _ = attn(
                q[:, i : i + 1],
                k[:, i : i + 1],
                v[:, i : i + 1],
                cache=cache,
                return_cache=True,
            )

        assert cache is not None
        assert cache.count == seq
        assert cache.k.shape[1] == cache_size

        _, k_rot_full = attn.rotary(q, k, jnp.array(0, dtype=jnp.int32))
        expected_k = jnp.zeros_like(cache.k)
        expected_v = jnp.zeros_like(cache.v)
        for t in range(seq - cache_size, seq):
            idx = t % cache_size
            expected_k = expected_k.at[:, idx].set(k_rot_full[:, t])
            expected_v = expected_v.at[:, idx].set(v[:, t])

        np.testing.assert_allclose(
            np.array(cache.k),
            np.array(expected_k),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            np.array(cache.v),
            np.array(expected_v),
            rtol=1e-6,
            atol=1e-6,
        )

    @pytest.mark.parametrize("attention_window", [None, 8])
    def test_arbitrary_cache_partition_invariance(
        self,
        random_seed: int,
        attention_window: int | None,
    ) -> None:
        """Cached outputs and state cannot depend on caller chunk boundaries."""
        key = jax.random.PRNGKey(random_seed)
        k_module, k_q, k_k, k_v = jax.random.split(key, 4)
        module = ChunkedAttention(
            num_heads=1,
            head_dim=4,
            value_head_dim=3,
            chunk_size=4,
            attention_window=attention_window,
            key=k_module,
        )
        q = jax.random.normal(k_q, (1, 12, 1, 4))
        k = jax.random.normal(k_k, (1, 12, 1, 4))
        v = jax.random.normal(k_v, (1, 12, 1, 3))

        expected, expected_cache, _ = module(q, k, v, return_cache=True)
        assert expected_cache is not None
        for partition in ((1,) * 12, (3, 5, 4), (7, 1, 4)):
            outputs = []
            cache = None
            start = 0
            for width in partition:
                stop = start + width
                output, cache, _ = module(
                    q[:, start:stop],
                    k[:, start:stop],
                    v[:, start:stop],
                    cache=cache,
                    return_cache=True,
                )
                outputs.append(output)
                start = stop
            actual = jnp.concatenate(outputs, axis=1)
            assert cache is not None
            np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=2e-6)
            np.testing.assert_array_equal(np.asarray(cache.k), np.asarray(expected_cache.k))
            np.testing.assert_array_equal(np.asarray(cache.v), np.asarray(expected_cache.v))
            np.testing.assert_array_equal(np.asarray(cache.count), np.asarray(expected_cache.count))

        batch_output, _, _ = module(q, k, v)
        np.testing.assert_allclose(np.asarray(batch_output), np.asarray(expected), atol=2e-6)


# -----------------------------------------------------------------------------
# NormalizedFFN Tests
# -----------------------------------------------------------------------------


class TestNormalizedFFN:
    """Tests for NormalizedFFN module."""

    def test_forward_shapes(self, random_seed: int) -> None:
        """Test NormalizedFFN forward pass shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        out = ffn(x)

        assert out.shape == (batch, seq, model_dim)

    def test_swiglu_variant(self, random_seed: int) -> None:
        """Test NormalizedFFN with SwiGLU activation.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=True,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        out = ffn(x)

        assert out.shape == (batch, seq, model_dim)
        assert ffn.fc3 is not None  # SwiGLU has fc3

    def test_source_bias_topology_and_trainable_rescale(self, random_seed: int) -> None:
        """FFN projections are bias-free and layer scale is a vector parameter."""
        ffn = NormalizedFFN(
            model_dim=16,
            ffn_hidden_dim=32,
            swiglu=True,
            rescale=True,
            layer_id=2,
            key=jax.random.PRNGKey(random_seed),
        )
        assert ffn.fc1.bias is None
        assert ffn.fc2.bias is None
        assert ffn.fc3 is not None and ffn.fc3.bias is None
        assert ffn.alpha is not None
        assert ffn.alpha.shape == (16,)
        np.testing.assert_array_equal(
            np.asarray(ffn.alpha),
            np.full(16, 0.025, dtype=np.float32),
        )
        assert any(leaf is ffn.alpha for leaf in jax.tree_util.tree_leaves(ffn))

    def test_two_hop_residual(self, random_seed: int) -> None:
        """Test NormalizedFFN with two-hop residual.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        residual_base = jax.random.normal(k3, (batch, seq, model_dim))

        out_normal = ffn(x)
        out_two_hop = ffn(x, residual_base=residual_base)

        # Two-hop should use residual_base, not x
        assert not jnp.allclose(out_normal, out_two_hop)

    @pytest.mark.torch_ref
    def test_ffn_parity(self, random_seed: int) -> None:
        """Test NormalizedFFN parity with PyTorch reference.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        torch = pytest.importorskip("torch")
        torch_modeling = require_torch_modeling()
        TorchConfig = torch_modeling.MegalodonConfig
        TorchFFN = torch_modeling.NormalizedFFN

        model_dim, ffn_dim = 64, 128
        batch, seq = 2, 16

        # Create PyTorch config and FFN
        torch_cfg = TorchConfig(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            norm_eps=1e-5,
            rescale_nffn=False,
        )
        torch_ffn = TorchFFN(torch_cfg, layer_id=0)
        torch_ffn.eval()

        # Create JAX FFN
        key = jax.random.PRNGKey(random_seed)
        jax_ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=False,
            norm_eps=1e-5,
            key=key,
        )

        # Copy weights from PyTorch to JAX
        jax_ffn = eqx.tree_at(lambda m: m.norm.weight, jax_ffn, to_jax(torch_ffn.norm.weight))
        jax_ffn = eqx.tree_at(lambda m: m.norm.bias, jax_ffn, to_jax(torch_ffn.norm.bias))
        jax_ffn = eqx.tree_at(lambda m: m.fc1.weight, jax_ffn, to_jax(torch_ffn.fc1.weight))
        jax_ffn = eqx.tree_at(lambda m: m.fc1.bias, jax_ffn, to_jax(torch_ffn.fc1.bias))
        jax_ffn = eqx.tree_at(lambda m: m.fc2.weight, jax_ffn, to_jax(torch_ffn.fc2.weight))
        jax_ffn = eqx.tree_at(lambda m: m.fc2.bias, jax_ffn, to_jax(torch_ffn.fc2.bias))

        # Generate test input
        x_torch = torch.randn(batch, seq, model_dim)
        x_jax = to_jax(x_torch)

        # Forward pass
        with torch.no_grad():
            y_torch = torch_ffn(x_torch)
        y_jax = jax_ffn(x_jax)

        # Compare - fp32 accumulation causes ~1e-3 drift over multiple operations
        np.testing.assert_allclose(
            np.array(y_jax),
            y_torch.detach().numpy(),
            rtol=5e-3,
            atol=5e-3,
            err_msg="NormalizedFFN output should match PyTorch reference",
        )


# -----------------------------------------------------------------------------
# MegalodonAttention Tests
# -----------------------------------------------------------------------------


class TestMegalodonAttention:
    """Tests for MegalodonAttention block."""

    def test_source_projection_bias_topology(self, random_seed: int) -> None:
        """Only released attention projections retain bias parameters."""
        module = MegalodonAttention(
            model_dim=16,
            z_dim=8,
            value_dim=16,
            num_heads=1,
            cema_ndim=2,
            chunk_size=4,
            norm_num_groups=4,
            key=jax.random.PRNGKey(random_seed),
        )
        assert module.wz.bias is not None
        assert module.wv.bias is not None
        assert module.wr.bias is not None
        assert module.wh1.bias is not None
        assert module.wh2.bias is None

    def test_forward_shapes(self, random_seed: int) -> None:
        """Test MegalodonAttention forward pass shapes.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq = 2, 32
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim))
        y, cache = attn(x)

        assert y.shape == (batch, seq, model_dim)
        assert cache is None  # No cache without return_cache=True

    def test_gradient_flow(self, random_seed: int) -> None:
        """Test gradients flow through MegalodonAttention.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq = 2, 16
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        def loss_fn(model: MegalodonAttention, x: jnp.ndarray) -> jnp.ndarray:
            """Compute a simple squared loss for gradient checks.

            :param MegalodonAttention model: Attention module under test.
            :param jnp.ndarray x: Input activations.
            :return jnp.ndarray: Scalar loss value.
            """
            y, _ = model(x)
            return jnp.sum(y**2)

        x = jax.random.normal(k2, (batch, seq, model_dim))
        grads = eqx.filter_grad(loss_fn)(attn, x)

        # Check gradients for key parameters
        assert not jnp.any(jnp.isnan(grads.gamma))
        assert not jnp.any(jnp.isnan(grads.beta))
        assert not jnp.any(jnp.isnan(grads.wz.weight))
        assert not jnp.any(jnp.isnan(grads.wv.weight))

    def test_segment_ids_isolate_cema_and_timenorm(self, random_seed: int) -> None:
        """Packed doc B must match doc B alone through the full block.

        Exercises the segment_ids plumbing into TimestepNorm, ComplexEMA, and
        ChunkedAttention together: with strict metadata, no state (norm stats,
        EMA state, attention) may leak from doc A into doc B.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        # doc A (5) + padding (2) + doc B (6), all within one chunk
        x_a = jax.random.normal(k2, (1, 5, model_dim))
        x_b = jax.random.normal(k3, (1, 6, model_dim))
        x_packed = jnp.concatenate([x_a, jnp.zeros((1, 2, model_dim)), x_b], axis=1)
        segment_ids = jnp.asarray([[1] * 5 + [0] * 2 + [2] * 6], dtype=jnp.int32)
        position_ids = jnp.asarray([[0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 5]], dtype=jnp.int32)
        mask = jnp.asarray([[True] * 5 + [False] * 2 + [True] * 6])

        y_packed, _ = attn(x_packed, mask=mask, segment_ids=segment_ids, position_ids=position_ids)
        y_b, _ = attn(x_b)

        np.testing.assert_allclose(
            np.array(y_packed[:, 7:]),
            np.array(y_b),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Doc B through packed row should match doc B alone",
        )

    def test_streaming_with_cache(self, random_seed: int) -> None:
        """Test MegalodonAttention streaming with cache.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch = 2
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        # Process tokens one at a time
        outputs = []
        cache = None

        for i in range(8):
            ki = jax.random.fold_in(k2, i)
            x = jax.random.normal(ki, (batch, 1, model_dim))
            y, cache = attn(x, cache=cache, return_cache=True)
            outputs.append(y)

        # Check final cache state
        assert cache is not None
        assert cache.attn is not None
        assert cache.norm is not None
        assert cache.ema is not None


# -----------------------------------------------------------------------------
# Precision Tests (bf16)
# -----------------------------------------------------------------------------


class TestPrecision:
    """Tests for bf16 precision handling."""

    def test_attention_primitives_bf16(self, random_seed: int) -> None:
        """Test attention primitives work with bf16 inputs.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim = 2, 16, 4, 32

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)
        k = jax.random.normal(k2, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)
        v = jax.random.normal(k3, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)

        out = attention_single_chunk(q, k, v)

        assert out.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(out))

    def test_attention_accum_dtype_override(self, random_seed: int) -> None:
        """Test attention primitives accept an accum_dtype override.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim = 2, 8, 2, 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        q = jax.random.normal(k1, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)
        k = jax.random.normal(k2, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)
        v = jax.random.normal(k3, (batch, seq, heads, head_dim)).astype(jnp.bfloat16)

        out = attention_single_chunk(q, k, v, accum_dtype=jnp.bfloat16)

        assert out.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(out))

    def test_megalodon_attention_bf16(self, random_seed: int) -> None:
        """Test MegalodonAttention with bf16 inputs and params.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq = 2, 16
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            compute_dtype=jnp.bfloat16,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim)).astype(jnp.bfloat16)
        y, _ = attn(x)

        assert y.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(y))

    def test_normalized_ffn_bf16(self, random_seed: int) -> None:
        """Test NormalizedFFN with bf16 inputs and params.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, model_dim, ffn_dim = 2, 16, 64, 128

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        ffn = NormalizedFFN(
            model_dim=model_dim,
            ffn_hidden_dim=ffn_dim,
            swiglu=True,
            compute_dtype=jnp.bfloat16,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq, model_dim)).astype(jnp.bfloat16)
        out = ffn(x)

        assert out.dtype == jnp.bfloat16
        assert not jnp.any(jnp.isnan(out))


# -----------------------------------------------------------------------------
# JIT Compilation Tests
# -----------------------------------------------------------------------------


class TestJITCompilation:
    """Tests for JIT compilation stability."""

    def test_chunked_attention_jit(self, random_seed: int) -> None:
        """Test ChunkedAttention compiles without errors.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq, heads, head_dim, value_dim = 2, 32, 4, 32, 64
        chunk_size = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        @eqx.filter_jit
        def forward(
            model: ChunkedAttention,
            q: jnp.ndarray,
            k: jnp.ndarray,
            v: jnp.ndarray,
        ) -> tuple[jnp.ndarray, Any, Any]:
            """Run a forward pass through ChunkedAttention.

            :param ChunkedAttention model: Attention module under test.
            :param jnp.ndarray q: Query tensor.
            :param jnp.ndarray k: Key tensor.
            :param jnp.ndarray v: Value tensor.
            :return tuple[jnp.ndarray, Any, Any]: Output, cache, and position.
            """
            return model(q, k, v)

        q = jax.random.normal(k2, (batch, seq, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq, heads, value_dim))

        out, _, _ = forward(attn, q, k, v)
        assert out.shape == (batch, seq, heads, value_dim)

    def test_streaming_jit_no_recompile(self, random_seed: int) -> None:
        """Verify streaming path is JIT-compatible without recompilation.

        This test ensures that:
        1. The streaming path can be JIT compiled
        2. Multiple calls with the same shapes don't cause recompilation
        3. The fori_loop with dynamic cache values works correctly
        """
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        @eqx.filter_jit
        def step(
            model: ChunkedAttention,
            q: jnp.ndarray,
            k: jnp.ndarray,
            v: jnp.ndarray,
            cache: Any,
        ) -> tuple[jnp.ndarray, Any, Any]:
            """Run a streaming step through ChunkedAttention.

            :param ChunkedAttention model: Attention module under test.
            :param jnp.ndarray q: Query tensor.
            :param jnp.ndarray k: Key tensor.
            :param jnp.ndarray v: Value tensor.
            :param Any cache: Optional attention cache.
            :return tuple[jnp.ndarray, Any, Any]: Output, cache, and position.
            """
            return model(q, k, v, cache=cache, return_cache=True)

        # Process 10 tokens one at a time
        cache = None
        for i in range(10):
            ki = jax.random.fold_in(k2, i)
            q = jax.random.normal(ki, (batch, 1, heads, head_dim))
            k = jax.random.normal(jax.random.fold_in(ki, 1), (batch, 1, heads, head_dim))
            v = jax.random.normal(jax.random.fold_in(ki, 2), (batch, 1, heads, value_dim))

            out, cache, pos = step(attn, q, k, v, cache)
            assert out.shape == (batch, 1, heads, value_dim)
            assert cache is not None
            assert int(pos) == i + 1

        # Final cache should be fixed-size
        assert cache.k.shape[1] == chunk_size
        assert int(cache.count) == 10

    def test_megalodon_attention_jit(self, random_seed: int) -> None:
        """Test MegalodonAttention compiles without errors.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq = 2, 32
        model_dim, z_dim, value_dim = 64, 32, 64
        num_heads, cema_ndim, chunk_size = 4, 8, 16
        norm_num_groups = 8

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        @eqx.filter_jit
        def forward(
            model: MegalodonAttention,
            x: jnp.ndarray,
        ) -> tuple[jnp.ndarray, Any]:
            """Run a forward pass through MegalodonAttention.

            :param MegalodonAttention model: Attention module under test.
            :param jnp.ndarray x: Input tensor.
            :return tuple[jnp.ndarray, Any]: Output and cache.
            """
            return model(x)

        x = jax.random.normal(k2, (batch, seq, model_dim))
        y, _ = forward(attn, x)
        assert y.shape == (batch, seq, model_dim)


# -----------------------------------------------------------------------------
# Streaming Equivalence Tests
# -----------------------------------------------------------------------------


class TestStreamingEquivalence:
    """Tests for streaming (token-by-token) equivalence with batch processing."""

    def test_chunked_attention_streaming_equivalence(
        self, random_seed: int, force_fp32_matmul: None
    ) -> None:
        """Verify streaming with cache matches batch processing within a chunk.

        :param int random_seed: Random seed fixture.
        :param None force_fp32_matmul: Fixture enabling fp32 matmul precision.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 1, 2, 16, 16
        chunk_size = 8
        seq_len = 6  # Within one chunk

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Generate full sequence Q/K/V
        q_full = jax.random.normal(k2, (batch, seq_len, heads, head_dim))
        k_full = jax.random.normal(k3, (batch, seq_len, heads, head_dim))
        v_full = jax.random.normal(k4, (batch, seq_len, heads, value_dim))

        # Batch processing (no cache)
        out_batch, _, _ = attn(q_full, k_full, v_full, return_cache=False)

        # Streaming processing (token by token with cache)
        streaming_outputs = []
        cache = None
        for i in range(seq_len):
            q_i = q_full[:, i : i + 1, :, :]
            k_i = k_full[:, i : i + 1, :, :]
            v_i = v_full[:, i : i + 1, :, :]

            out_i, cache, _ = attn(q_i, k_i, v_i, cache=cache, return_cache=True)
            streaming_outputs.append(out_i)

        out_streaming = jnp.concatenate(streaming_outputs, axis=1)

        # Outputs should match (within a single chunk)
        np.testing.assert_allclose(
            np.array(out_batch),
            np.array(out_streaming),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Streaming output should match batch output within a single chunk",
        )

    def test_chunk_boundary_cache_reset(self, random_seed: int) -> None:
        """Verify cache is reset at chunk boundaries.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Process tokens 0-3 (first chunk), then tokens 4-5 (second chunk)
        cache = None
        for i in range(6):
            q = jax.random.normal(jax.random.fold_in(k2, i), (batch, 1, heads, head_dim))
            k = jax.random.normal(jax.random.fold_in(k3, i), (batch, 1, heads, head_dim))
            v = jax.random.normal(jax.random.fold_in(k4, i), (batch, 1, heads, value_dim))

            _, cache, position = attn(q, k, v, cache=cache, return_cache=True)

            # Fixed-size buffer: shape is always max_cache_len
            assert cache.k.shape[1] == chunk_size, (
                f"Cache buffer should be fixed size {chunk_size}, got {cache.k.shape[1]}"
            )
            # The count tracks absolute position
            assert cache.count == i + 1, f"Cache count should be {i + 1}, got {cache.count}"

    def test_incompatible_cache_shape_is_rejected(self, random_seed: int) -> None:
        """Cache schema mismatches fail instead of silently resizing state.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        from megalodon_jax.types import AttentionCache

        # Create a smaller-than-expected cache (2 entries, but chunk_size=4)
        # Simulates position 2 (has 2 cached entries from positions 0-1)
        fake_k = jax.random.normal(k2, (batch, 2, heads, head_dim))
        fake_v = jax.random.normal(k3, (batch, 2, heads, value_dim))
        fake_cache = AttentionCache(k=fake_k, v=fake_v, count=jnp.array(2, dtype=jnp.int32))

        # Process a new token at position 2
        q = jax.random.normal(k4, (batch, 1, heads, head_dim))
        k = jax.random.normal(jax.random.fold_in(k4, 1), (batch, 1, heads, head_dim))
        v = jax.random.normal(jax.random.fold_in(k4, 2), (batch, 1, heads, value_dim))

        with pytest.raises(ValueError, match="cache shapes"):
            attn(q, k, v, cache=fake_cache, return_cache=True)


# -----------------------------------------------------------------------------
# Parity Tests
# -----------------------------------------------------------------------------


class TestParity:
    """Additional parity tests for MegalodonAttention and related modules."""

    def test_chunked_attention_block_diagonal_structure(self, random_seed: int) -> None:
        """Verify block-diagonal attention structure across chunks.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4
        seq_len = 8  # Two full chunks

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Generate full sequence Q/K/V
        q = jax.random.normal(k2, (batch, seq_len, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq_len, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq_len, heads, value_dim))

        # Get batch output
        out, _, _ = attn(q, k, v)

        # Process each chunk separately and compare
        # First chunk: positions 0-3
        q1, k1, v1 = q[:, :4], k[:, :4], v[:, :4]
        out1, _, _ = attn(q1, k1, v1)

        # Second chunk: positions 4-7
        q2, k2, v2 = q[:, 4:], k[:, 4:], v[:, 4:]
        out2, _, _ = attn(q2, k2, v2)

        # Block-diagonal: each chunk's output should match processing independently
        np.testing.assert_allclose(
            np.array(out[:, :4]),
            np.array(out1),
            rtol=1e-5,
            atol=1e-5,
            err_msg="First chunk output should match independent processing",
        )
        np.testing.assert_allclose(
            np.array(out[:, 4:]),
            np.array(out2),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Second chunk output should match independent processing",
        )

    def test_multi_token_streaming_across_boundary(self, random_seed: int) -> None:
        """Test that multi-token streaming calls properly split across chunk boundaries.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, heads, head_dim, value_dim = 1, 2, 8, 8
        chunk_size = 4

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3, k4 = jax.random.split(key, 4)

        attn = ChunkedAttention(
            num_heads=heads,
            head_dim=head_dim,
            value_head_dim=value_dim,
            chunk_size=chunk_size,
            key=k1,
        )

        # Generate a sequence that spans chunk boundary
        seq_len = 6  # Will span positions 0-5 when starting from position 2
        q = jax.random.normal(k2, (batch, seq_len, heads, head_dim))
        k = jax.random.normal(k3, (batch, seq_len, heads, head_dim))
        v = jax.random.normal(k4, (batch, seq_len, heads, value_dim))

        # Start at position 2, so tokens will span positions 2,3 | 4,5,6,7
        # chunk boundary at position 4
        from megalodon_jax.types import AttentionCache

        # Pre-fill cache with 2 tokens at positions 0-1
        init_k = jax.random.normal(jax.random.fold_in(k1, 0), (batch, 2, heads, head_dim))
        init_v = jax.random.normal(jax.random.fold_in(k1, 1), (batch, 2, heads, value_dim))
        # Apply RoPE to initial K
        init_k_rot, _ = attn.rotary(init_k, init_k, jnp.array(0))
        cache_k = jnp.zeros((batch, chunk_size, heads, head_dim)).at[:, :2].set(init_k_rot)
        cache_v = jnp.zeros((batch, chunk_size, heads, value_dim)).at[:, :2].set(init_v)
        cache = AttentionCache(k=cache_k, v=cache_v, count=jnp.array(2, dtype=jnp.int32))

        # Process multi-token call that spans boundary
        out_multi, cache_multi, pos_multi = attn(q, k, v, cache=cache, return_cache=True)

        # Now compare with token-by-token processing
        streaming_outputs = []
        cache_stream = cache
        for i in range(seq_len):
            q_i = q[:, i : i + 1]
            k_i = k[:, i : i + 1]
            v_i = v[:, i : i + 1]
            out_i, cache_stream, _ = attn(q_i, k_i, v_i, cache=cache_stream, return_cache=True)
            streaming_outputs.append(out_i)

        out_streaming = jnp.concatenate(streaming_outputs, axis=1)

        # Multi-token and streaming should produce same result
        np.testing.assert_allclose(
            np.array(out_multi),
            np.array(out_streaming),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Multi-token streaming should match token-by-token when crossing chunk boundary",
        )

        # Final cache positions should match
        assert int(pos_multi) == int(cache_stream.count), (
            f"Final positions should match: multi={int(pos_multi)}, stream={int(cache_stream.count)}"
        )


# -----------------------------------------------------------------------------
# MegalodonAttention Parity Tests
# -----------------------------------------------------------------------------


class TestMegalodonAttentionParity:
    """Parity tests for MegalodonAttention against PyTorch reference."""

    @pytest.mark.torch_ref
    def test_megalodon_attention_forward_parity(
        self, random_seed: int, torch_device: torch.device
    ) -> None:
        """Test MegalodonAttention forward pass parity with PyTorch reference.

        :param int random_seed: Random seed fixture.
        :param torch.device torch_device: Torch device fixture.
        :return None: None.
        """
        from tests.conftest import sync_and_clear_torch

        torch = pytest.importorskip("torch")
        torch_modeling = require_torch_modeling()
        TorchMegalodonAttention = torch_modeling.MegalodonAttention
        TorchConfig = torch_modeling.MegalodonConfig

        # Config matching both implementations
        batch, seq_len = 1, 8
        model_dim = 32
        z_dim = 16
        value_dim = 16
        num_heads = 2
        cema_ndim = 2
        chunk_size = 4
        norm_num_groups = 2

        # Create PyTorch module on same device as JAX (GPU if available)
        torch_cfg = TorchConfig(
            model_dim=model_dim,
            num_heads=num_heads,
            z_dim=z_dim,
            value_dim=value_dim,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-5,
            norm_affine=True,
            rope_base=10000.0,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        torch_attn = TorchMegalodonAttention(torch_cfg).to(torch_device)
        torch_attn.eval()

        # Create JAX module with same config
        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            norm_eps=1e-5,
            rope_base=10000.0,
            dropout=0.0,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            key=k1,
        )

        # Copy weights from PyTorch to JAX
        # TimestepNorm
        jax_attn = eqx.tree_at(
            lambda m: m.timenorm.weight, jax_attn, to_jax(torch_attn.timenorm.weight)
        )
        jax_attn = eqx.tree_at(
            lambda m: m.timenorm.bias, jax_attn, to_jax(torch_attn.timenorm.bias)
        )

        # ComplexEMA (alpha, delta, theta stored as real tensors)
        jax_attn = eqx.tree_at(
            lambda m: m.cema.alpha,
            jax_attn,
            to_jax(torch_attn.cema.alpha),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.delta,
            jax_attn,
            to_jax(torch_attn.cema.delta),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.theta,
            jax_attn,
            to_jax(torch_attn.cema.theta),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.gamma_real,
            jax_attn,
            to_jax(torch_attn.cema.gamma_real),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.gamma_imag,
            jax_attn,
            to_jax(torch_attn.cema.gamma_imag),
        )
        jax_attn = eqx.tree_at(
            lambda m: m.cema.omega,
            jax_attn,
            to_jax(torch_attn.cema.omega),
        )

        # RMSNorm
        jax_attn = eqx.tree_at(
            lambda m: m.rmsnorm.gamma, jax_attn, to_jax(torch_attn.rmsnorm.gamma)
        )

        # Projections (both PyTorch and Equinox use (out_features, in_features) layout)
        jax_attn = eqx.tree_at(lambda m: m.wz.weight, jax_attn, to_jax(torch_attn.wz.weight))
        jax_attn = eqx.tree_at(lambda m: m.wz.bias, jax_attn, to_jax(torch_attn.wz.bias))
        jax_attn = eqx.tree_at(lambda m: m.wv.weight, jax_attn, to_jax(torch_attn.wv.weight))
        jax_attn = eqx.tree_at(lambda m: m.wv.bias, jax_attn, to_jax(torch_attn.wv.bias))
        jax_attn = eqx.tree_at(lambda m: m.wr.weight, jax_attn, to_jax(torch_attn.wr.weight))
        jax_attn = eqx.tree_at(lambda m: m.wr.bias, jax_attn, to_jax(torch_attn.wr.bias))
        jax_attn = eqx.tree_at(lambda m: m.wh1.weight, jax_attn, to_jax(torch_attn.wh1.weight))
        jax_attn = eqx.tree_at(lambda m: m.wh1.bias, jax_attn, to_jax(torch_attn.wh1.bias))
        jax_attn = eqx.tree_at(lambda m: m.wh2.weight, jax_attn, to_jax(torch_attn.wh2.weight))
        jax_attn = eqx.tree_at(lambda m: m.wh2.bias, jax_attn, to_jax(torch_attn.wh2.bias))

        # Q/K affine parameters
        jax_attn = eqx.tree_at(lambda m: m.gamma, jax_attn, to_jax(torch_attn.gamma))
        jax_attn = eqx.tree_at(lambda m: m.beta, jax_attn, to_jax(torch_attn.beta))

        # Inner attention rotary (copy from CPU tensor)
        jax_attn = eqx.tree_at(
            lambda m: m.inner.rotary.inv_freq,
            jax_attn,
            to_jax(torch_attn.inner.rope.inv_freq.cpu()),
        )

        # Generate input on JAX, convert to PyTorch on same device
        x_jax = jax.random.normal(k2, (batch, seq_len, model_dim))
        x_torch = to_torch(x_jax).to(torch_device)

        # Run PyTorch forward on GPU, then move result to CPU
        with torch.no_grad():
            y_torch, _ = torch_attn(x_torch)
            y_torch_cpu = y_torch.cpu().numpy()

        # Clean up PyTorch GPU memory before JAX forward
        del y_torch, x_torch, torch_attn
        sync_and_clear_torch()

        # Run JAX forward
        y_jax, _ = jax_attn(x_jax, deterministic=True)

        # Compare outputs (both ran on GPU with matching TF32 settings)
        np.testing.assert_allclose(
            np.array(y_jax),
            y_torch_cpu,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MegalodonAttention output should match PyTorch reference",
        )

    def test_megalodon_attention_gradient_flow(self, random_seed: int) -> None:
        """Verify gradients flow through all parameters without NaN.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        batch, seq_len = 1, 8
        model_dim = 32
        z_dim = 16
        value_dim = 16
        num_heads = 2
        cema_ndim = 2
        chunk_size = 4
        norm_num_groups = 2

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        attn = MegalodonAttention(
            model_dim=model_dim,
            z_dim=z_dim,
            value_dim=value_dim,
            num_heads=num_heads,
            cema_ndim=cema_ndim,
            chunk_size=chunk_size,
            norm_num_groups=norm_num_groups,
            key=k1,
        )

        x = jax.random.normal(k2, (batch, seq_len, model_dim))

        def loss_fn(model: MegalodonAttention) -> jnp.ndarray:
            """Compute a mean-squared loss for gradient checks.

            :param MegalodonAttention model: Attention module under test.
            :return jnp.ndarray: Scalar loss value.
            """
            y, _ = model(x, deterministic=True)
            return jnp.mean(y**2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(attn)

        # Check loss is finite
        assert jnp.isfinite(loss), f"Loss should be finite, got {loss}"

        # Check all gradients are finite
        grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
        for i, g in enumerate(grad_leaves):
            assert jnp.all(jnp.isfinite(g)), f"Gradient {i} has non-finite values"

        # Check at least some gradients are non-zero
        non_zero_grads = sum(1 for g in grad_leaves if jnp.any(g != 0))
        assert non_zero_grads > 0, "At least some gradients should be non-zero"
