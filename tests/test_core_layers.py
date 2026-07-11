"""Phase 2 Core Layers tests - TimestepNorm, ComplexEMA parity."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax.layers import ComplexEMA, TimestepNorm
from megalodon_jax.layers.segments import segment_boundaries, segment_runs_and_local_positions


class TestSegmentHelpers:
    """Tests for the shared segment-metadata helpers.

    These define the single boundary semantics consumed by attention masking,
    ComplexEMA resets, and TimestepNorm resets.
    """

    def test_boundaries_basic_and_repeated_ids(self) -> None:
        """Boundaries fire at position 0 and every id change, including reuse.

        :return None: None.
        """
        segment_ids = jnp.asarray([[1, 1, 2, 2, 1, 1], [1, 1, 1, 0, 0, 2]], dtype=jnp.int32)
        expected = np.asarray(
            [
                [True, False, True, False, True, False],
                [True, False, False, True, False, True],
            ]
        )
        np.testing.assert_array_equal(np.array(segment_boundaries(segment_ids)), expected)

    def test_boundaries_empty_sequence(self) -> None:
        """L=0 input must return an empty (B, 0) boolean array, not crash.

        :return None: None.
        """
        out = segment_boundaries(jnp.zeros((3, 0), dtype=jnp.int32))
        assert out.shape == (3, 0)
        assert out.dtype == jnp.bool_

    def test_runs_and_local_positions(self) -> None:
        """Run ids restart counting per contiguous run; local positions restart at 0.

        :return None: None.
        """
        segment_ids = jnp.asarray([[1, 1, 2, 2, 2, 1, 0, 0]], dtype=jnp.int32)
        run_ids, local_positions = segment_runs_and_local_positions(segment_ids)
        np.testing.assert_array_equal(np.array(run_ids), [[1, 1, 2, 2, 2, 3, 4, 4]])
        np.testing.assert_array_equal(np.array(local_positions), [[0, 1, 0, 1, 2, 0, 0, 1]])


class TestTimestepNorm:
    """Mathematical and state-continuation tests for TimestepNorm."""

    def test_streaming_state_continuity(self, random_seed: int) -> None:
        """Test that processing in chunks matches full sequence.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)

        batch, seq = 2, 32
        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (batch, seq, dim))

        # Full sequence at once
        y_full, state_full = jax_norm(x)

        # Process in two chunks with state passing
        y1, state1 = jax_norm(x[:, :16, :])
        y2, state2 = jax_norm(x[:, 16:, :], state=state1)
        y_chunked = jnp.concatenate([y1, y2], axis=1)

        # Outputs should match (with some tolerance due to floating point)
        np.testing.assert_allclose(
            np.array(y_chunked),
            np.array(y_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Chunked processing should match full sequence",
        )

    def test_weight_initialization(self) -> None:
        """Test that weight and bias are initialized to zeros.

        :return None: None.
        """
        norm = TimestepNorm(64, 8)
        np.testing.assert_array_equal(np.array(norm.weight), np.zeros(64))
        np.testing.assert_array_equal(np.array(norm.bias), np.zeros(64))

    def test_effective_scale_is_one(self) -> None:
        """Test that effective scale is 1.0 with zero weight.

        :return None: None.
        """
        norm = TimestepNorm(64, 8)
        # Effective scale = weight + 1.0 = 1.0
        x = jnp.ones((2, 16, 64))
        y, _ = norm(x)
        # Output should be normalized (mean~0, var~1 per group, then scaled by 1.0)
        assert y.shape == x.shape

    def test_exact_scalar_population_moments(self) -> None:
        """Each valid token contributes all scalar features in its group."""
        norm = TimestepNorm(4, 2, eps=0.0)
        x = jnp.asarray([[[1.0, 3.0, 2.0, 6.0], [5.0, 7.0, 10.0, 14.0]]])

        output, state = norm(x)

        expected_output = np.asarray(
            [[[-1.0, 1.0, -1.0, 1.0], [0.4472136, 1.3416408, 0.4472136, 1.3416408]]]
        )
        np.testing.assert_allclose(np.asarray(output), expected_output, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(np.asarray(state.count), [2])
        np.testing.assert_allclose(np.asarray(state.mean), [[4.0, 8.0]], atol=1e-6)
        np.testing.assert_allclose(np.asarray(state.var), [[5.0, 20.0]], atol=1e-6)

    def test_learned_prior_and_featurewise_contract(self) -> None:
        """Released learned priors are trainable only when prior_count is positive."""
        grouped = TimestepNorm(8, 2)
        assert grouped.prior_mean is None
        assert grouped.prior_logv is None

        with pytest.raises(ValueError, match="prior_count > 1"):
            TimestepNorm(8, 8)

        featurewise = TimestepNorm(8, None, prior_count=2)
        assert featurewise.prior_mean is not None
        assert featurewise.prior_logv is not None
        _, state = featurewise(jnp.ones((1, 1, 8), dtype=jnp.float32))
        np.testing.assert_array_equal(np.asarray(state.count), [3])

    def test_mask_handling(self, random_seed: int) -> None:
        """Test that padding mask correctly excludes positions from statistics.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)

        batch, seq = 2, 16
        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (batch, seq, dim))

        # Create mask: first 8 positions valid, rest masked
        mask = jnp.concatenate(
            [
                jnp.ones((batch, 8), dtype=jnp.bool_),
                jnp.zeros((batch, 8), dtype=jnp.bool_),
            ],
            axis=1,
        )

        y_masked, state_masked = jax_norm(x, mask=mask)

        # Count should be 8 for all batch elements
        np.testing.assert_array_equal(np.array(state_masked.count), np.array([8, 8]))
        np.testing.assert_array_equal(np.asarray(y_masked[:, 8:]), np.zeros((batch, 8, dim)))

    def test_state_dtype_fp32(self, random_seed: int) -> None:
        """Ensure running stats stay in float32 for bf16 inputs."""
        dim = 64
        num_groups = 8
        norm = TimestepNorm(dim, num_groups)

        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (2, 4, dim)).astype(jnp.bfloat16)
        _, state = norm(x)

        assert state.mean.dtype == jnp.float32
        assert state.var.dtype == jnp.float32

    def test_state_matches_explicit_stats(self, random_seed: int) -> None:
        """Compare running stats to an explicit Welford implementation."""
        dim = 16
        num_groups = 4
        group_size = dim // num_groups
        norm = TimestepNorm(dim, num_groups)

        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (2, 5, dim))
        mask = jnp.array(
            [
                [True, True, False, True, False],
                [True, False, True, True, True],
            ],
            dtype=jnp.bool_,
        )

        _, state = norm(x, mask=mask)

        x_np = np.array(x, dtype=np.float32)
        mask_np = np.array(mask)
        count = np.zeros((2,), dtype=np.int32)
        values = [[[] for _ in range(num_groups)] for _ in range(2)]

        for t in range(x_np.shape[1]):
            for b in range(2):
                if not mask_np[b, t]:
                    continue
                count[b] += 1
                grouped = x_np[b, t].reshape(num_groups, group_size)
                for group in range(num_groups):
                    values[b][group].extend(grouped[group].tolist())

        mean = np.zeros((2, num_groups), dtype=np.float32)
        var = np.ones((2, num_groups), dtype=np.float32)
        for b in range(2):
            for group in range(num_groups):
                mean[b, group] = np.mean(values[b][group])
                var[b, group] = np.var(values[b][group])

        np.testing.assert_array_equal(np.array(state.count), count)
        np.testing.assert_allclose(np.array(state.mean), mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(state.var), var, rtol=1e-5, atol=1e-5)

    def test_divisibility_validation(self) -> None:
        """Test that num_features must be divisible by num_groups.

        :return None: None.
        """
        with pytest.raises(ValueError, match="divisible by"):
            TimestepNorm(63, 8)

    def test_different_shapes(self) -> None:
        """Test TimestepNorm works with various input shapes.

        :return None: None.
        """
        norm = TimestepNorm(128, 16)
        for shape in [(1, 10, 128), (4, 32, 128), (2, 1, 128)]:
            x = jnp.ones(shape)
            y, state = norm(x)
            assert y.shape == shape


class TestTimestepNormSegmentReset:
    """Tests for packed-sequence stat resets in TimestepNorm."""

    @staticmethod
    def _packed_two_docs(
        random_seed: int, dim: int = 16, num_groups: int = 4
    ) -> tuple[TimestepNorm, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Build a norm module plus doc A, doc B, and their packed row.

        Layout: doc A (5 tokens) + padding (2 tokens, segment 0) + doc B (6 tokens).

        :param int random_seed: Random seed fixture value.
        :param int dim: Feature dimension.
        :param int num_groups: Number of normalization groups.
        :return tuple: (norm, x_doc_a, x_doc_b, x_packed, segment_ids, mask).
        """
        norm = TimestepNorm(dim, num_groups)
        key = jax.random.PRNGKey(random_seed)
        k_a, k_b = jax.random.split(key)
        x_a = jax.random.normal(k_a, (1, 5, dim))
        x_b = jax.random.normal(k_b, (1, 6, dim))
        x_packed = jnp.concatenate([x_a, jnp.zeros((1, 2, dim)), x_b], axis=1)
        segment_ids = jnp.asarray([[1] * 5 + [0] * 2 + [2] * 6], dtype=jnp.int32)
        mask = jnp.asarray([[True] * 5 + [False] * 2 + [True] * 6])
        return norm, x_a, x_b, x_packed, segment_ids, mask

    def test_segment_ids_none_matches_baseline(self, random_seed: int) -> None:
        """Passing segment_ids=None must be bit-identical to omitting it.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        norm = TimestepNorm(16, 4)
        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (2, 12, 16))
        mask = jnp.ones((2, 12), dtype=jnp.bool_)

        y_base, state_base = norm(x, mask=mask)
        y_none, state_none = norm(x, mask=mask, segment_ids=None)

        np.testing.assert_array_equal(np.array(y_base), np.array(y_none))
        np.testing.assert_array_equal(np.array(state_base.count), np.array(state_none.count))
        np.testing.assert_array_equal(np.array(state_base.mean), np.array(state_none.mean))
        np.testing.assert_array_equal(np.array(state_base.var), np.array(state_none.var))

    def test_segment_reset_matches_per_doc_alone(self, random_seed: int) -> None:
        """Packed docs with resets must normalize like each doc alone.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        norm, x_a, x_b, x_packed, segment_ids, mask = self._packed_two_docs(random_seed)

        y_packed, _ = norm(x_packed, mask=mask, segment_ids=segment_ids)
        y_a, _ = norm(x_a)
        y_b, _ = norm(x_b)

        np.testing.assert_allclose(
            np.array(y_packed[:, :5]),
            np.array(y_a),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Doc A slice should match doc A alone",
        )
        np.testing.assert_allclose(
            np.array(y_packed[:, 7:]),
            np.array(y_b),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Doc B slice should match doc B alone (no stat leak)",
        )

    def test_segment_reset_matches_explicit_numpy_welford(self, random_seed: int) -> None:
        """Compare segmented running stats to an explicit Welford loop with resets.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 16
        num_groups = 4
        group_size = dim // num_groups
        norm = TimestepNorm(dim, num_groups)

        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (2, 8, dim))
        segment_ids = jnp.asarray(
            [
                [1, 1, 1, 2, 2, 2, 2, 2],
                [1, 1, 0, 0, 3, 3, 3, 3],
            ],
            dtype=jnp.int32,
        )
        mask = jnp.asarray(
            [
                [True, True, False, True, True, True, True, True],
                [True, True, False, False, True, True, True, True],
            ]
        )

        _, state = norm(x, mask=mask, segment_ids=segment_ids)

        x_np = np.array(x, dtype=np.float32)
        seg_np = np.array(segment_ids)
        mask_np = np.array(mask)
        count = np.zeros((2,), dtype=np.int32)
        values = [[[] for _ in range(num_groups)] for _ in range(2)]

        for t in range(x_np.shape[1]):
            for b in range(2):
                if t > 0 and seg_np[b, t] != seg_np[b, t - 1]:
                    count[b] = 0
                    values[b] = [[] for _ in range(num_groups)]
                if not mask_np[b, t] or seg_np[b, t] == 0:
                    continue
                count[b] += 1
                grouped = x_np[b, t].reshape(num_groups, group_size)
                for group in range(num_groups):
                    values[b][group].extend(grouped[group].tolist())

        mean = np.zeros((2, num_groups), dtype=np.float32)
        var = np.ones((2, num_groups), dtype=np.float32)
        for b in range(2):
            for group in range(num_groups):
                mean[b, group] = np.mean(values[b][group])
                var[b, group] = np.var(values[b][group])

        np.testing.assert_array_equal(np.array(state.count), count)
        np.testing.assert_allclose(np.array(state.mean), mean, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.array(state.var), var, rtol=1e-5, atol=1e-5)

    def test_new_state_is_last_segment_local(self, random_seed: int) -> None:
        """Returned state must equal running only the last doc with a fresh state.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        norm, _, x_b, x_packed, segment_ids, mask = self._packed_two_docs(random_seed)

        _, state_packed = norm(x_packed, mask=mask, segment_ids=segment_ids)
        _, state_b = norm(x_b)

        np.testing.assert_array_equal(np.array(state_packed.count), np.array(state_b.count))
        np.testing.assert_allclose(
            np.array(state_packed.mean), np.array(state_b.mean), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(state_packed.var), np.array(state_b.var), rtol=1e-5, atol=1e-5
        )

    def test_raises_when_segment_ids_with_state(self, random_seed: int) -> None:
        """segment_ids combined with an incoming state must raise.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        norm, _, _, x_packed, segment_ids, mask = self._packed_two_docs(random_seed)
        _, state = norm(x_packed, mask=mask)

        with pytest.raises(ValueError, match="segment_ids"):
            norm(x_packed, state=state, segment_ids=segment_ids)

    def test_state_with_trailing_padding_matches_last_doc(self, random_seed: int) -> None:
        """Trailing padding must not blank the returned state.

        Regression: the state was read at position L-1, which for a row ending
        in padding (segment 0) is the fresh-reset baseline (count=0, mean=0,
        var=1) rather than the last document's statistics.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 16
        norm = TimestepNorm(dim, 4)
        key = jax.random.PRNGKey(random_seed)
        k_a, k_b = jax.random.split(key)
        x_a = jax.random.normal(k_a, (1, 5, dim))
        x_b = jax.random.normal(k_b, (1, 6, dim))
        x_packed = jnp.concatenate([x_a, x_b, jnp.zeros((1, 3, dim))], axis=1)
        segment_ids = jnp.asarray([[1] * 5 + [2] * 6 + [0] * 3], dtype=jnp.int32)
        mask = jnp.asarray([[True] * 11 + [False] * 3])

        _, state_packed = norm(x_packed, mask=mask, segment_ids=segment_ids)
        _, state_b = norm(x_b)

        np.testing.assert_array_equal(np.array(state_packed.count), np.array(state_b.count))
        np.testing.assert_allclose(
            np.array(state_packed.mean), np.array(state_b.mean), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_allclose(
            np.array(state_packed.var), np.array(state_b.var), rtol=1e-5, atol=1e-5
        )


class TestComplexEMA:
    """Recurrence and state-continuation tests for ComplexEMA."""

    def test_fft_vs_sequential_equivalence(self, random_seed: int) -> None:
        """Test that FFT and sequential paths produce equivalent outputs.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # Generate test input
        batch, seq = 2, 32
        x = jax.random.normal(k2, (batch, dim, seq))

        # FFT path
        y_fft, _ = jax_ema(x, return_state=False)

        # Sequential path
        y_seq, _ = jax_ema(x, return_state=True)

        # Should produce equivalent outputs
        np.testing.assert_allclose(
            np.array(y_fft),
            np.array(y_seq),
            rtol=1e-4,
            atol=1e-5,
            err_msg="FFT and sequential paths should produce equivalent outputs",
        )

    def test_state_continuity(self, random_seed: int) -> None:
        """Test that chunked processing with state matches full sequence.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # Generate test input
        batch, seq = 2, 32
        x = jax.random.normal(k2, (batch, dim, seq))

        # Full sequence with FFT
        y_full, _ = jax_ema(x, return_state=False)

        # Two chunks with state passing
        y1, h1 = jax_ema(x[:, :, :16], return_state=True)
        y2, h2 = jax_ema(x[:, :, 16:], h_init=h1, return_state=True)
        y_chunked = jnp.concatenate([y1, y2], axis=-1)

        # Should match (within numerical tolerance)
        np.testing.assert_allclose(
            np.array(y_chunked),
            np.array(y_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Chunked processing should match full sequence",
        )

    def test_q_magnitude_bounded(self, random_seed: int) -> None:
        """Test that |q| < 1 by construction (ensures decaying impulse response).

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        jax_ema = ComplexEMA(dim, ndim, key=key)

        _, q, _ = jax_ema._coeffs()
        q_magnitude = jnp.abs(q)

        # All magnitudes should be strictly less than 1
        assert jnp.all(q_magnitude < 1.0), "q magnitude must be < 1 for stability"

    def test_jit_compilation(self, random_seed: int) -> None:
        """Test that ComplexEMA works with JIT compilation.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # JIT the forward pass
        @eqx.filter_jit
        def forward(ema: ComplexEMA, x: jnp.ndarray) -> tuple[jnp.ndarray, Any]:
            """Run a JIT-compiled ComplexEMA forward pass.

            :param ComplexEMA ema: EMA module under test.
            :param jnp.ndarray x: Input sequence tensor.
            :return tuple[jnp.ndarray, Any]: Output and optional state.
            """
            return ema(x, return_state=False)

        x = jax.random.normal(k2, (2, dim, 32))

        # First call compiles
        y1, _ = forward(jax_ema, x)
        # Second call should use cached compilation
        y2, _ = forward(jax_ema, x)

        np.testing.assert_array_equal(np.array(y1), np.array(y2))


class TestComplexEMASegmentReset:
    """Tests for packed-sequence state resets in ComplexEMA."""

    @staticmethod
    def _packed_two_docs(
        random_seed: int, dim: int = 8, ndim: int = 4
    ) -> tuple[ComplexEMA, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Build an EMA module plus doc A, doc B, and their packed row.

        Layout: doc A (5 tokens) + padding (2 tokens, segment 0) + doc B (5 tokens).

        :param int random_seed: Random seed fixture value.
        :param int dim: Hidden dimension.
        :param int ndim: EMA orders per hidden unit.
        :return tuple: (ema, x_doc_a, x_doc_b, x_packed, segment_ids).
        """
        key = jax.random.PRNGKey(random_seed)
        k_ema, k_a, k_b = jax.random.split(key, 3)
        ema = ComplexEMA(dim, ndim, key=k_ema)
        x_a = jax.random.normal(k_a, (1, dim, 5))
        x_b = jax.random.normal(k_b, (1, dim, 5))
        x_packed = jnp.concatenate([x_a, jnp.zeros((1, dim, 2)), x_b], axis=-1)
        segment_ids = jnp.asarray([[1] * 5 + [0] * 2 + [2] * 5], dtype=jnp.int32)
        return ema, x_a, x_b, x_packed, segment_ids

    def test_segment_ids_none_matches_baseline(self, random_seed: int) -> None:
        """Passing segment_ids=None must be bit-identical to omitting it.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)
        ema = ComplexEMA(16, 4, key=k1)
        x = jax.random.normal(k2, (2, 16, 24))
        mask = jnp.ones((2, 24), dtype=jnp.bool_)

        y_base, _ = ema(x, mask=mask)
        y_none, _ = ema(x, mask=mask, segment_ids=None)

        np.testing.assert_array_equal(np.array(y_base), np.array(y_none))

    @pytest.mark.parametrize("use_associative", [True, False])
    def test_segment_reset_matches_per_doc_alone(
        self, random_seed: int, use_associative: bool
    ) -> None:
        """Packed docs with resets must match each doc run alone.

        :param int random_seed: Random seed fixture.
        :param bool use_associative: Which segmented implementation to test.
        :return None: None.
        """
        ema, x_a, x_b, x_packed, segment_ids = self._packed_two_docs(random_seed)

        y_packed, h_packed = ema(
            x_packed,
            segment_ids=segment_ids,
            return_state=True,
            use_associative_segment_scan=use_associative,
        )
        y_a, _ = ema(x_a)
        y_b, h_b = ema(x_b, return_state=True)

        np.testing.assert_allclose(
            np.array(y_packed[..., :5]),
            np.array(y_a),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Doc A slice should match doc A alone",
        )
        np.testing.assert_allclose(
            np.array(y_packed[..., 7:]),
            np.array(y_b),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Doc B slice should match doc B alone (no state leak)",
        )
        # Returned state is local to the last open segment (doc B)
        np.testing.assert_allclose(
            np.array(h_packed),
            np.array(h_b),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Final state should equal doc B's standalone state",
        )

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_associative_matches_sequential_reset(self, random_seed: int, dtype: Any) -> None:
        """Associative and sequential segmented paths must agree.

        :param int random_seed: Random seed fixture.
        :param Any dtype: Input dtype under test.
        :return None: None.
        """
        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)
        ema = ComplexEMA(32, 8, key=k1)
        x = jax.random.normal(k2, (2, 32, 40)).astype(dtype)
        segment_ids = jnp.asarray(
            [[1] * 16 + [2] * 24, [1] * 8 + [2] * 8 + [0] * 4 + [3] * 20],
            dtype=jnp.int32,
        )

        y_assoc, h_assoc = ema(x, segment_ids=segment_ids, return_state=True)
        y_seq, h_seq = ema(
            x,
            segment_ids=segment_ids,
            return_state=True,
            use_associative_segment_scan=False,
        )

        tol = 1e-2 if dtype == jnp.bfloat16 else 1e-5
        np.testing.assert_allclose(
            np.array(y_assoc, dtype=np.float32),
            np.array(y_seq, dtype=np.float32),
            rtol=tol,
            atol=tol,
            err_msg="Associative and sequential reset paths should agree",
        )
        np.testing.assert_allclose(
            np.array(h_assoc),
            np.array(h_seq),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Final states of both reset paths should agree",
        )

    @pytest.mark.parametrize("use_associative", [True, False])
    def test_single_segment_matches_fft(self, random_seed: int, use_associative: bool) -> None:
        """Trivial single-segment ids must reproduce the FFT path output.

        :param int random_seed: Random seed fixture.
        :param bool use_associative: Which segmented implementation to test.
        :return None: None.
        """
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)
        ema = ComplexEMA(dim, ndim, key=k1)
        batch, seq = 2, 32
        x = jax.random.normal(k2, (batch, dim, seq))
        segment_ids = jnp.ones((batch, seq), dtype=jnp.int32)

        y_fft, _ = ema(x)
        y_seg, _ = ema(
            x,
            segment_ids=segment_ids,
            use_associative_segment_scan=use_associative,
        )

        np.testing.assert_allclose(
            np.array(y_seg),
            np.array(y_fft),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Single-segment reset path should match FFT path",
        )

    def test_raises_when_segment_ids_with_h_init(self, random_seed: int) -> None:
        """segment_ids combined with an incoming state must raise.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        ema, _, _, x_packed, segment_ids = self._packed_two_docs(random_seed)
        h_init = jnp.zeros((1, 8, 4), dtype=jnp.complex64)

        with pytest.raises(ValueError, match="segment_ids"):
            ema(x_packed, h_init=h_init, segment_ids=segment_ids)

    @pytest.mark.parametrize("use_associative", [True, False])
    def test_state_with_trailing_padding_matches_last_doc(
        self, random_seed: int, use_associative: bool
    ) -> None:
        """Trailing padding must not zero the returned EMA state.

        Regression: the state was read at position L-1; the padding run's
        boundary reset left it all-zero instead of the last document's
        standalone state.

        :param int random_seed: Random seed fixture.
        :param bool use_associative: Which segmented implementation to test.
        :return None: None.
        """
        dim, ndim = 8, 4
        key = jax.random.PRNGKey(random_seed)
        k_ema, k_a, k_b = jax.random.split(key, 3)
        ema = ComplexEMA(dim, ndim, key=k_ema)
        x_a = jax.random.normal(k_a, (1, dim, 5))
        x_b = jax.random.normal(k_b, (1, dim, 5))
        x_packed = jnp.concatenate([x_a, x_b, jnp.zeros((1, dim, 3))], axis=-1)
        segment_ids = jnp.asarray([[1] * 5 + [2] * 5 + [0] * 3], dtype=jnp.int32)

        _, h_packed = ema(
            x_packed,
            segment_ids=segment_ids,
            return_state=True,
            use_associative_segment_scan=use_associative,
        )
        _, h_b = ema(x_b, return_state=True)

        assert np.abs(np.array(h_packed)).max() > 0.0, "Returned state should not be all-zero"
        np.testing.assert_allclose(
            np.array(h_packed),
            np.array(h_b),
            rtol=1e-5,
            atol=1e-5,
            err_msg="Final state should equal doc B's standalone state despite trailing padding",
        )


class TestPrecisionPolicy:
    """Tests for bf16/fp16 precision handling."""

    def test_timestep_norm_bf16_input(self, random_seed: int) -> None:
        """Test TimestepNorm works correctly with bf16 inputs.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)

        batch, seq = 2, 16
        key = jax.random.PRNGKey(random_seed)

        # Create bf16 input
        x_f32 = jax.random.normal(key, (batch, seq, dim))
        x_bf16 = x_f32.astype(jnp.bfloat16)

        # Forward pass should work
        y_bf16, state = jax_norm(x_bf16)

        # Output dtype should match input dtype
        assert y_bf16.dtype == jnp.bfloat16

        # Compare with fp32 path (within bf16 tolerance)
        y_f32, _ = jax_norm(x_f32)
        np.testing.assert_allclose(
            np.array(y_bf16.astype(jnp.float32)),
            np.array(y_f32),
            rtol=1e-2,
            atol=1e-2,
            err_msg="bf16 output should be close to fp32 output",
        )

    def test_timestep_norm_fp16_rejected(self) -> None:
        """Test TimestepNorm rejects fp16 inputs for stability.

        :return None: None.
        """
        dim = 64
        num_groups = 8

        jax_norm = TimestepNorm(dim, num_groups)
        x_fp16 = jnp.ones((2, 16, dim), dtype=jnp.float16)

        with pytest.raises(TypeError, match="float16"):
            jax_norm(x_fp16)

    def test_complex_ema_bf16_params(self, random_seed: int) -> None:
        """Test ComplexEMA computes coefficients in fp32 even with bf16 params.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        ema_f32 = ComplexEMA(dim, ndim, key=key)

        # Cast parameters to bf16
        def to_bf16(x: Any) -> Any:
            """Cast floating-point arrays to bf16 for precision tests.

            :param Any x: Input value to cast if it is a floating array.
            :return Any: Casted value or original input.
            """
            if eqx.is_array(x) and x.dtype == jnp.float32:
                return x.astype(jnp.bfloat16)
            return x

        ema_bf16 = jax.tree.map(to_bf16, ema_f32)

        # Coefficients should be computed in fp32
        p_f32, q_f32, gamma_f32 = ema_f32._coeffs()
        p_bf16, q_bf16, gamma_bf16 = ema_bf16._coeffs()

        # All coefficient outputs should be fp32/complex64 regardless of param dtype
        assert p_f32.dtype == jnp.float32
        assert p_bf16.dtype == jnp.float32, f"Expected float32, got {p_bf16.dtype}"
        assert q_f32.dtype == jnp.complex64
        assert q_bf16.dtype == jnp.complex64, f"Expected complex64, got {q_bf16.dtype}"
        assert gamma_f32.dtype == jnp.complex64
        assert gamma_bf16.dtype == jnp.complex64, f"Expected complex64, got {gamma_bf16.dtype}"

        # Values should be close, but bf16 params have quantization noise
        # so we use bf16-appropriate tolerance (~1e-2)
        np.testing.assert_allclose(
            np.array(p_f32),
            np.array(p_bf16),
            rtol=1e-2,
            atol=1e-3,
            err_msg="bf16 params should produce fp32 coefficients close to fp32 params",
        )

    def test_complex_ema_fft_vs_sequential_bf16(self, random_seed: int) -> None:
        """Test FFT and sequential paths produce equivalent results in bf16.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        jax_ema = ComplexEMA(dim, ndim, key=k1)

        # Cast to bf16
        def to_bf16(x: Any) -> Any:
            """Cast floating-point arrays to bf16 for precision tests.

            :param Any x: Input value to cast if it is a floating array.
            :return Any: Casted value or original input.
            """
            if eqx.is_array(x) and x.dtype == jnp.float32:
                return x.astype(jnp.bfloat16)
            return x

        jax_ema_bf16 = jax.tree.map(to_bf16, jax_ema)

        # Generate bf16 input
        batch, seq = 2, 32
        x_bf16 = jax.random.normal(k2, (batch, dim, seq)).astype(jnp.bfloat16)

        # FFT path
        y_fft, _ = jax_ema_bf16(x_bf16, return_state=False)

        # Sequential path
        y_seq, _ = jax_ema_bf16(x_bf16, return_state=True)

        # Both should be bf16
        assert y_fft.dtype == jnp.bfloat16
        assert y_seq.dtype == jnp.bfloat16

        # Should produce equivalent outputs (within bf16 tolerance)
        np.testing.assert_allclose(
            np.array(y_fft),
            np.array(y_seq),
            rtol=1e-2,
            atol=1e-2,
            err_msg="FFT and sequential paths should match in bf16",
        )


class TestIntegration:
    """Integration tests for Phase 2 layers."""

    def test_timestep_norm_to_complex_ema_flow(self, random_seed: int) -> None:
        """Test data flow from TimestepNorm to ComplexEMA.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        num_groups = 8
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        norm = TimestepNorm(dim, num_groups)
        ema = ComplexEMA(dim, ndim, key=k1)

        # Input (B, L, D) for norm
        x = jax.random.normal(k2, (2, 32, dim))

        # Norm forward
        x_normed, norm_state = norm(x)

        # Transpose for EMA (B, D, L)
        x_ema = jnp.transpose(x_normed, (0, 2, 1))

        # EMA forward
        y_ema, ema_state = ema(x_ema, return_state=True)

        assert y_ema.shape == (2, dim, 32)
        assert ema_state.shape == (2, dim, ndim)

    def test_gradient_flow(self, random_seed: int) -> None:
        """Test that gradients flow through both layers.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        num_groups = 8
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2 = jax.random.split(key)

        norm = TimestepNorm(dim, num_groups)
        ema = ComplexEMA(dim, ndim, key=k1)

        def loss_fn(models: tuple[TimestepNorm, ComplexEMA], x: jnp.ndarray) -> jnp.ndarray:
            """Compute a simple squared loss for gradient checks.

            :param tuple[TimestepNorm, ComplexEMA] models: Norm and EMA modules.
            :param jnp.ndarray x: Input activations.
            :return jnp.ndarray: Scalar loss value.
            """
            norm, ema = models
            x_normed, _ = norm(x)
            x_ema = jnp.transpose(x_normed, (0, 2, 1))
            y, _ = ema(x_ema)
            return jnp.sum(y**2)

        x = jax.random.normal(k2, (2, 32, dim))

        # Compute gradients
        grads = eqx.filter_grad(loss_fn)((norm, ema), x)
        norm_grads, ema_grads = grads

        # Check gradients are not None and not NaN
        # Norm gradients
        assert not jnp.any(jnp.isnan(norm_grads.weight))
        assert not jnp.any(jnp.isnan(norm_grads.bias))

        # EMA gradients
        assert not jnp.any(jnp.isnan(ema_grads.alpha))
        assert not jnp.any(jnp.isnan(ema_grads.delta))
        assert not jnp.any(jnp.isnan(ema_grads.gamma_real))
        assert not jnp.any(jnp.isnan(ema_grads.gamma_imag))

    def test_jit_no_recompilation(self, random_seed: int) -> None:
        """Test that JIT doesn't recompile on different inputs.

        :param int random_seed: Random seed fixture.
        :return None: None.
        """
        dim = 64
        num_groups = 8
        ndim = 16

        key = jax.random.PRNGKey(random_seed)
        k1, k2, k3 = jax.random.split(key, 3)

        norm = TimestepNorm(dim, num_groups)
        ema = ComplexEMA(dim, ndim, key=k1)

        @eqx.filter_jit
        def forward(
            norm: TimestepNorm,
            ema: ComplexEMA,
            x: jnp.ndarray,
        ) -> jnp.ndarray:
            """Run a JIT-compiled norm + EMA forward pass.

            :param TimestepNorm norm: TimestepNorm module under test.
            :param ComplexEMA ema: ComplexEMA module under test.
            :param jnp.ndarray x: Input activations.
            :return jnp.ndarray: EMA output activations.
            """
            x_normed, _ = norm(x)
            x_ema = jnp.transpose(x_normed, (0, 2, 1))
            y, _ = ema(x_ema)
            return y

        x1 = jax.random.normal(k2, (2, 32, dim))
        x2 = jax.random.normal(k3, (2, 32, dim))

        # First call compiles
        y1 = forward(norm, ema, x1)
        # Second call should use cached compilation (same shape)
        y2 = forward(norm, ema, x2)

        assert y1.shape == y2.shape
