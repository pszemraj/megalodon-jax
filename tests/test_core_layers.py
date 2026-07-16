"""Phase 2 Core Layers tests - TimestepNorm, ComplexEMA parity."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax.layers import ComplexEMA, TimestepNorm
from megalodon_jax.layers.segments import (
    derive_segment_metadata,
    segment_boundaries,
    segment_runs_and_local_positions,
)
from megalodon_jax.layers.timestep_norm import (
    _block_moments,
    _merge_m2,
    _MomentSummary,
    _shifted_cumsum_prefix,
)
from megalodon_jax.types import NormState
from tests.factories import floating_to_bf16
from tests.reference.timestep_norm import PaperNormState, central_difference, timestep_norm_paper


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

    def test_derived_metadata_matches_individual_helpers(self) -> None:
        """The shared pytree preserves every established packed-sequence predicate."""
        segment_ids = jnp.asarray(
            [[1, 1, 2, 2, 1, 0], [3, 3, 3, 4, 4, 4]],
            dtype=jnp.int32,
        )
        metadata = derive_segment_metadata(segment_ids)
        run_ids, local_positions = segment_runs_and_local_positions(segment_ids)

        np.testing.assert_array_equal(metadata.valid, segment_ids > 0)
        np.testing.assert_array_equal(metadata.boundaries, segment_boundaries(segment_ids))
        np.testing.assert_array_equal(metadata.run_ids, run_ids)
        np.testing.assert_array_equal(metadata.local_positions, local_positions)


class TestTimestepNorm:
    """Mathematical and state-continuation tests for TimestepNorm."""

    def test_chan_m2_merge_and_zero_identities(self) -> None:
        """Compact summaries use exact identities and unnormalized M2."""
        left = _MomentSummary(
            count=jnp.asarray([[[2]]], dtype=jnp.int32),
            mean=jnp.asarray([[[3.0]]]),
            m2=jnp.asarray([[[4.0]]]),
        )
        right = _MomentSummary(
            count=jnp.asarray([[[3]]], dtype=jnp.int32),
            mean=jnp.asarray([[[7.0]]]),
            m2=jnp.asarray([[[6.0]]]),
        )
        identity = _MomentSummary(
            count=jnp.zeros((1, 1, 1), dtype=jnp.int32),
            mean=jnp.zeros((1, 1, 1)),
            m2=jnp.zeros((1, 1, 1)),
        )

        merged = _merge_m2(left, right)

        np.testing.assert_array_equal(np.asarray(merged.count), [[[5]]])
        np.testing.assert_allclose(np.asarray(merged.mean), [[[5.4]]], atol=1e-6)
        np.testing.assert_allclose(np.asarray(merged.m2), [[[29.2]]], atol=1e-6)
        for expected, actual in zip(left, _merge_m2(identity, left), strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        for expected, actual in zip(left, _merge_m2(left, identity), strict=True):
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))

    @pytest.mark.fast
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
        y_full, _ = jax_norm(x)

        # Process in two chunks with state passing
        y1, state1 = jax_norm(x[:, :16, :])
        y2, _ = jax_norm(x[:, 16:, :], state=state1)
        y_chunked = jnp.concatenate([y1, y2], axis=1)

        # Outputs should match (with some tolerance due to floating point)
        np.testing.assert_allclose(
            np.array(y_chunked),
            np.array(y_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Chunked processing should match full sequence",
        )

    def test_masked_streaming_state_continuity(self, random_seed: int) -> None:
        """Masked associative prefixes continue identically across calls."""
        norm = TimestepNorm(32, 4)
        key = jax.random.PRNGKey(random_seed)
        x = jax.random.normal(key, (2, 17, 32), dtype=jnp.float32)
        mask = (jnp.arange(17)[None, :] + jnp.arange(2)[:, None]) % 4 != 1

        expected, expected_state = norm(x, mask=mask)
        first, state = norm(x[:, :7], mask=mask[:, :7])
        second, actual_state = norm(x[:, 7:], state=state, mask=mask[:, 7:])
        actual = jnp.concatenate((first, second), axis=1)

        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(
            np.asarray(actual_state.count), np.asarray(expected_state.count)
        )
        np.testing.assert_allclose(
            np.asarray(actual_state.mean), np.asarray(expected_state.mean), rtol=2e-5, atol=2e-5
        )
        np.testing.assert_allclose(
            np.asarray(actual_state.var), np.asarray(expected_state.var), rtol=2e-5, atol=2e-5
        )

    def test_weight_initialization(self) -> None:
        """Test that weight and bias are initialized to zeros.

        :return None: None.
        """
        norm = TimestepNorm(64, 8)
        np.testing.assert_array_equal(np.array(norm.weight), np.zeros(64))
        np.testing.assert_array_equal(np.array(norm.bias), np.zeros(64))

    @pytest.mark.fast
    def test_exact_scalar_population_moments(self) -> None:
        """Each valid token contributes all scalar features in its group."""
        norm = TimestepNorm(4, 2, eps=1e-12)
        x = jnp.asarray([[[1.0, 3.0, 2.0, 6.0], [5.0, 7.0, 10.0, 14.0]]])

        output, state = norm(x)

        expected_output = np.asarray(
            [[[-1.0, 1.0, -1.0, 1.0], [0.4472136, 1.3416408, 0.4472136, 1.3416408]]]
        )
        np.testing.assert_allclose(np.asarray(output), expected_output, atol=2e-6, rtol=2e-6)
        np.testing.assert_array_equal(np.asarray(state.count), [2])

        constant_output, _ = TimestepNorm(4, 2)(jnp.ones((1, 2, 4), dtype=jnp.float32))
        np.testing.assert_array_equal(np.asarray(constant_output), np.zeros((1, 2, 4)))
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

    @pytest.mark.fast
    def test_fully_masked_row_preserves_incoming_state(self) -> None:
        """A row containing only identity summaries returns its state exactly."""
        norm = TimestepNorm(8, 2)
        state = NormState(
            count=jnp.asarray([7], dtype=jnp.int32),
            mean=jnp.asarray([[2.0, -3.0]], dtype=jnp.float32),
            var=jnp.asarray([[0.5, 4.0]], dtype=jnp.float32),
        )
        x = jnp.full((1, 5, 8), jnp.nan, dtype=jnp.float32)

        output, result = norm(x, state=state, mask=jnp.zeros((1, 5), dtype=jnp.bool_))

        np.testing.assert_array_equal(np.asarray(output), np.zeros((1, 5, 8)))
        for name in ("count", "mean", "var"):
            np.testing.assert_array_equal(
                np.asarray(getattr(result, name)),
                np.asarray(getattr(state, name)),
            )

    @pytest.mark.fast
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

    def test_large_offset_constant_has_zero_variance(self) -> None:
        """Stable Welford prefixes must not invent variance from a large mean."""
        norm = TimestepNorm(32, 4)
        x = jnp.full((1, 512, 32), 1e7, dtype=jnp.float32)

        output, state = norm(x)

        np.testing.assert_array_equal(np.asarray(output), np.zeros(x.shape, dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(state.mean), np.full((1, 4), 1e7))
        np.testing.assert_array_equal(np.asarray(state.var), np.zeros((1, 4)))

    def test_shifted_hot_path_all_prefix_variances_are_nonnegative(self) -> None:
        """Adversarial finite inputs retain finite nonnegative prefix variance."""
        norm = TimestepNorm(32, 4)
        key = jax.random.PRNGKey(31)
        normal = jax.random.normal(key, (2, 257, 32), dtype=jnp.float32)
        time = jnp.arange(257, dtype=jnp.float32)[None, :, None]
        alternating = jnp.broadcast_to(
            jnp.where(
                (jnp.arange(257)[None, :, None] + jnp.arange(32)[None, None]) % 2,
                -1.0,
                1.0,
            ),
            normal.shape,
        )
        families = (
            normal,
            37.0 + 1.7 * normal,
            jnp.full_like(normal, 7.25),
            3.0 + 1e-4 * normal,
            alternating,
            0.075 * time + 0.02 * normal,
            1e2 + 2e-4 * normal,
            1e4 + 2e-2 * normal,
            1e6 + 2.0 * normal,
            (11.0 + 2.0 * normal).astype(jnp.bfloat16),
        )
        states = (
            norm._prior_state(2),  # noqa: SLF001 - direct hot-path invariant test.
            NormState(
                count=jnp.asarray([7, 11], dtype=jnp.int32),
                mean=jnp.asarray([[8.0] * 4, [-3.0] * 4], dtype=jnp.float32),
                var=jnp.asarray([[0.25] * 4, [2.0] * 4], dtype=jnp.float32),
            ),
        )

        for values in families:
            grouped = values.astype(jnp.float32).reshape(2, 257, 4, 8)
            block_mean, block_var = _block_moments(grouped)
            for initial in states:
                _, _, prefix_var = _shifted_cumsum_prefix(block_mean, block_var, initial)
                assert bool(jnp.all(jnp.isfinite(prefix_var)))
                assert float(jnp.min(prefix_var)) >= 0.0

    def test_unmasked_continuation_uses_stable_moment_merge(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Continuation history bypasses cancellation-prone cumsum moments."""
        norm = TimestepNorm(8, 2)
        state = NormState(
            count=jnp.asarray([1], dtype=jnp.int32),
            mean=jnp.zeros((1, 2), dtype=jnp.float32),
            var=jnp.zeros((1, 2), dtype=jnp.float32),
        )
        values = jnp.full((1, 257, 8), 1e7, dtype=jnp.float32)

        def fail_shifted_prefix(*args: object, **kwargs: object) -> None:
            pytest.fail("continuations must use associative moment merging")

        monkeypatch.setattr(
            "megalodon_jax.layers.timestep_norm._shifted_cumsum_prefix",
            fail_shifted_prefix,
        )
        output, result = norm(values, state=state)

        assert bool(jnp.all(jnp.isfinite(output)))
        assert bool(jnp.all(jnp.isfinite(result.var)))
        assert float(jnp.min(result.var)) >= 0.0

    def test_masked_associative_path_remains_finite_at_large_offsets(self) -> None:
        """Masked Welford prefixes retain finite, nonnegative state and output."""
        norm = TimestepNorm(32, 4)
        key = jax.random.PRNGKey(47)
        noise = jax.random.normal(key, (2, 257, 32), dtype=jnp.float32)
        mask = (jnp.arange(257)[None, :] + jnp.arange(2)[:, None]) % 5 != 2

        for values in (
            jnp.full_like(noise, 1e7),
            1e6 + 2.0 * noise,
            1e4 + 2e-2 * noise,
            (1e3 + 2.0 * noise).astype(jnp.bfloat16),
        ):
            output, state = norm(values, mask=mask)
            assert bool(jnp.all(jnp.isfinite(output)))
            assert bool(jnp.all(jnp.isfinite(state.mean)))
            assert bool(jnp.all(jnp.isfinite(state.var)))
            assert float(jnp.min(state.var)) >= 0.0

    @pytest.mark.fast
    @pytest.mark.parametrize("mode", ["plain", "masked", "packed", "continuation"])
    def test_matches_float64_paper_oracle(self, mode: str) -> None:
        """Production paths agree with an independent scalar paper equation."""
        norm = TimestepNorm(8, 2, eps=1e-5)
        weight = jnp.linspace(-0.15, 0.2, 8, dtype=jnp.float32)
        bias = jnp.linspace(0.07, -0.05, 8, dtype=jnp.float32)
        norm = eqx.tree_at(lambda item: item.weight, norm, weight)
        norm = eqx.tree_at(lambda item: item.bias, norm, bias)
        x = jax.random.normal(jax.random.PRNGKey(73), (2, 7, 8), dtype=jnp.float32) * 1.7 + 11.0
        state = None
        paper_state = None
        mask = None
        segment_ids = None
        if mode == "masked":
            mask = jnp.asarray(
                [
                    [True, False, True, True, False, True, True],
                    [False, True, True, False, True, True, False],
                ]
            )
        elif mode == "packed":
            segment_ids = jnp.asarray(
                [[0, 1, 1, 2, 2, 1, 0], [3, 3, 0, 4, 4, 4, 5]],
                dtype=jnp.int32,
            )
        elif mode == "continuation":
            state = NormState(
                count=jnp.asarray([7, 11], dtype=jnp.int32),
                mean=jnp.asarray([[8.0, 9.0], [-4.0, 2.0]], dtype=jnp.float32),
                var=jnp.asarray([[0.25, 1.5], [3.0, 0.5]], dtype=jnp.float32),
            )
            paper_state = PaperNormState(
                count=np.asarray(state.count, dtype=np.int64),
                mean=np.asarray(state.mean, dtype=np.float64),
                var=np.asarray(state.var, dtype=np.float64),
            )

        output, final = norm(x, state=state, mask=mask, segment_ids=segment_ids)
        expected_output, expected_final, valid = timestep_norm_paper(
            np.asarray(x, dtype=np.float64),
            groups=2,
            eps=norm.eps,
            weight=np.asarray(weight),
            bias=np.asarray(bias),
            state=paper_state,
            mask=None if mask is None else np.asarray(mask),
            segment_ids=None if segment_ids is None else np.asarray(segment_ids),
        )

        np.testing.assert_allclose(np.asarray(output), expected_output, rtol=3e-5, atol=3e-5)
        np.testing.assert_array_equal(np.asarray(final.count), expected_final.count)
        np.testing.assert_allclose(np.asarray(final.mean), expected_final.mean, atol=3e-5)
        np.testing.assert_allclose(np.asarray(final.var), expected_final.var, atol=3e-5)
        np.testing.assert_array_equal(np.asarray(output)[~valid], 0.0)

    @pytest.mark.fast
    def test_large_offset_continuation_matches_oracle_and_gradients(self) -> None:
        """Large-count shifted continuation remains stable and differentiable."""
        norm = TimestepNorm(4, 2, eps=1e-5)
        x = jnp.asarray(
            [
                [
                    [1_000_000.125, 999_999.875, 1_000_000.25, 999_999.75],
                    [999_999.75, 1_000_000.25, 999_999.875, 1_000_000.125],
                    [1_000_000.25, 1_000_000.0, 999_999.75, 1_000_000.0],
                ]
            ],
            dtype=jnp.float32,
        )
        weight = jnp.asarray([-0.2, 0.1, 0.25, -0.05], dtype=jnp.float32)
        bias = jnp.asarray([0.03, -0.07, 0.11, -0.02], dtype=jnp.float32)
        mean = jnp.asarray([[999_999.75, 1_000_000.5]], dtype=jnp.float32)
        var = jnp.asarray([[0.015625, 0.0625]], dtype=jnp.float32)
        count = jnp.asarray([1_000_003], dtype=jnp.int32)
        output_coefficients = jnp.linspace(-0.4, 0.7, x.size, dtype=jnp.float32).reshape(x.shape)
        mean_coefficients = jnp.asarray([[0.17, -0.09]], dtype=jnp.float32)
        var_coefficients = jnp.asarray([[-0.11, 0.13]], dtype=jnp.float32)

        def objective(
            values: jax.Array,
            stored_weight: jax.Array,
            stored_bias: jax.Array,
            state_mean: jax.Array,
            state_var: jax.Array,
        ) -> jax.Array:
            module = eqx.tree_at(lambda item: item.weight, norm, stored_weight)
            module = eqx.tree_at(lambda item: item.bias, module, stored_bias)
            result, final = module(
                values,
                state=NormState(count=count, mean=state_mean, var=state_var),
            )
            return (
                jnp.sum(result * output_coefficients)
                + jnp.sum(final.mean * mean_coefficients)
                + jnp.sum(final.var * var_coefficients)
            )

        value, gradients = jax.value_and_grad(objective, argnums=(0, 1, 2, 3, 4))(
            x, weight, bias, mean, var
        )
        module = eqx.tree_at(lambda item: item.weight, norm, weight)
        module = eqx.tree_at(lambda item: item.bias, module, bias)
        output, final = module(x, state=NormState(count=count, mean=mean, var=var))
        grouped = x.reshape(1, 3, 2, 2)
        block_mean, block_var = _block_moments(grouped)
        _, _, prefix_var = _shifted_cumsum_prefix(
            block_mean,
            block_var,
            NormState(count=count, mean=mean, var=var),
        )

        def paper_objective(
            values: np.ndarray,
            stored_weight: np.ndarray,
            stored_bias: np.ndarray,
            state_mean: np.ndarray,
            state_var: np.ndarray,
        ) -> float:
            result, paper_final, _ = timestep_norm_paper(
                values,
                groups=2,
                eps=norm.eps,
                weight=stored_weight,
                bias=stored_bias,
                state=PaperNormState(
                    count=np.asarray(count, dtype=np.int64),
                    mean=state_mean,
                    var=state_var,
                ),
            )
            return float(
                np.sum(result * np.asarray(output_coefficients, dtype=np.float64))
                + np.sum(paper_final.mean * np.asarray(mean_coefficients, dtype=np.float64))
                + np.sum(paper_final.var * np.asarray(var_coefficients, dtype=np.float64))
            )

        arrays = tuple(np.asarray(item, dtype=np.float64) for item in (x, weight, bias, mean, var))
        steps = (0.125, 1e-3, 1e-3, 0.125, 1e-3)
        expected_gradients = []
        for argument, (array, step) in enumerate(zip(arrays, steps, strict=True)):
            expected_gradients.append(
                central_difference(
                    lambda candidate, argument=argument: paper_objective(
                        *(
                            candidate if index == argument else value
                            for index, value in enumerate(arrays)
                        )
                    ),
                    array,
                    step,
                )
            )

        expected_output, expected_final, _ = timestep_norm_paper(
            arrays[0],
            groups=2,
            eps=norm.eps,
            weight=arrays[1],
            bias=arrays[2],
            state=PaperNormState(
                count=np.asarray(count, dtype=np.int64),
                mean=arrays[3],
                var=arrays[4],
            ),
        )
        assert bool(jnp.isfinite(value))
        assert bool(jnp.all(jnp.isfinite(prefix_var)))
        assert float(jnp.min(prefix_var)) >= 0.0
        np.testing.assert_allclose(np.asarray(output), expected_output, rtol=2e-2, atol=6e-2)
        np.testing.assert_array_equal(np.asarray(final.count), expected_final.count)
        np.testing.assert_allclose(np.asarray(final.mean), expected_final.mean, atol=6.25e-2)
        np.testing.assert_allclose(np.asarray(final.var), expected_final.var, rtol=2e-3, atol=2e-4)
        for actual, expected in zip(gradients, expected_gradients, strict=True):
            np.testing.assert_allclose(np.asarray(actual), expected, rtol=2e-2, atol=5e-3)

    @pytest.mark.fast
    @pytest.mark.parametrize("mode", ["plain", "masked", "packed"])
    def test_forward_and_backward_have_no_sequence_while(self, mode: str) -> None:
        """Production sequence work remains outside runtime WhileOps."""
        norm = TimestepNorm(32, 4)
        x = jnp.ones((1, 65, 32), dtype=jnp.float32)
        mask = jnp.arange(65)[None] % 5 != 2 if mode == "masked" else None
        segment_ids = None
        if mode == "packed":
            segment_ids = jnp.where(
                jnp.arange(65)[None] % 11 == 0,
                0,
                jnp.arange(65)[None] // 13 + 1,
            ).astype(jnp.int32)

        def loss(values: jax.Array) -> jax.Array:
            output, state = norm(values, mask=mask, segment_ids=segment_ids)
            return (
                jnp.sum(jnp.sin(output.astype(jnp.float32)))
                + 1e-3 * jnp.sum(state.mean)
                + 1e-4 * jnp.sum(state.var)
            )

        functions = {
            "forward": jax.jit(lambda values: norm(values, mask=mask, segment_ids=segment_ids)),
            "forward_backward": jax.jit(jax.value_and_grad(loss)),
        }
        for name, function in functions.items():
            stablehlo = str(function.lower(x).compiler_ir(dialect="stablehlo"))
            assert "stablehlo.while" not in stablehlo, f"{mode} {name}"
            dynamic_updates = [
                line for line in stablehlo.splitlines() if "stablehlo.dynamic_update_slice" in line
            ]
            if name == "forward":
                assert not dynamic_updates, mode
            else:
                assert all("tensor<1x65x4x8xf32>" not in line for line in dynamic_updates), mode

    def test_masked_nonfinite_token_does_not_update_state(self) -> None:
        """Masked NaN/Inf blocks cannot contaminate later valid prefixes."""
        norm = TimestepNorm(4, 2, eps=1e-12)
        x = jnp.asarray(
            [
                [
                    [1.0, 3.0, 2.0, 6.0],
                    [jnp.nan, jnp.inf, -jnp.inf, jnp.nan],
                    [5.0, 7.0, 10.0, 14.0],
                ]
            ],
            dtype=jnp.float32,
        )
        mask = jnp.asarray([[True, False, True]])

        output, state = norm(x, mask=mask)

        expected = np.asarray(
            [
                [
                    [-1.0, 1.0, -1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.4472136, 1.3416408, 0.4472136, 1.3416408],
                ]
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(np.asarray(output), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(np.asarray(state.count), [2])
        np.testing.assert_allclose(np.asarray(state.mean), [[4.0, 8.0]], atol=1e-6)
        np.testing.assert_allclose(np.asarray(state.var), [[5.0, 20.0]], atol=1e-6)

    @pytest.mark.fast
    def test_masked_nonfinite_tokens_have_zero_finite_gradients(self) -> None:
        """Inactive NaN/Inf payloads cannot poison reverse-mode arithmetic."""
        norm = TimestepNorm(8, 2)
        valid = jnp.asarray(
            [
                [False, True, True, True],
                [True, False, True, True],
                [True, True, True, False],
                [False, False, False, False],
            ],
            dtype=jnp.bool_,
        )
        finite = jax.random.normal(jax.random.PRNGKey(37), (4, 4, 8))
        nonfinite = jnp.asarray(
            [jnp.nan, jnp.inf, -jnp.inf, jnp.nan, jnp.inf, -jnp.inf, jnp.nan, jnp.inf]
        )
        x = jnp.where(valid[..., None], finite, nonfinite)

        def objective(values: jax.Array) -> jax.Array:
            output, state = norm(values, mask=valid)
            return (
                jnp.sum(jnp.square(output)) + 0.1 * jnp.sum(state.mean) + 0.03 * jnp.sum(state.var)
            )

        value, gradient = jax.value_and_grad(objective)(x)

        assert bool(jnp.isfinite(value))
        assert bool(jnp.all(jnp.isfinite(gradient)))
        np.testing.assert_array_equal(
            np.asarray(gradient)[~np.asarray(valid)],
            np.zeros((7, 8), dtype=np.float32),
        )

    def test_prior_count_overflow_is_rejected(self) -> None:
        """A fresh learned-prior state cannot wrap its int32 count."""
        norm = TimestepNorm(4, 2, prior_count=int(jnp.iinfo(jnp.int32).max))
        with pytest.raises(ValueError, match="overflow"):
            norm(jnp.ones((1, 1, 4), dtype=jnp.float32))

    @pytest.mark.parametrize(
        ("count", "mean", "var"),
        [
            pytest.param(-1, 0.0, 1.0, id="negative-count"),
            pytest.param(int(jnp.iinfo(jnp.int32).max), 0.0, 1.0, id="count-overflow"),
            pytest.param(0, float("nan"), 1.0, id="nonfinite-mean"),
            pytest.param(0, 0.0, float("inf"), id="nonfinite-variance"),
            pytest.param(0, 0.0, -1.0, id="negative-variance"),
        ],
    )
    def test_continuation_state_rejects_invalid_values(
        self,
        count: int,
        mean: float,
        var: float,
    ) -> None:
        """Malformed or exhausted continuation state fails before prefix work."""
        norm = TimestepNorm(4, 2)
        state = NormState(
            count=jnp.asarray([count], dtype=jnp.int32),
            mean=jnp.full((1, 2), mean, dtype=jnp.float32),
            var=jnp.full((1, 2), var, dtype=jnp.float32),
        )

        with pytest.raises(Exception, match="TimestepNorm state"):
            norm(jnp.ones((1, 1, 4), dtype=jnp.float32), state=state)

    def test_continuation_count_boundary_and_masked_identity(self) -> None:
        """The last int32 count is reachable and a masked call cannot advance it."""
        norm = TimestepNorm(4, 2)
        maximum = int(jnp.iinfo(jnp.int32).max)
        state = NormState(
            count=jnp.asarray([maximum - 1], dtype=jnp.int32),
            mean=jnp.zeros((1, 2), dtype=jnp.float32),
            var=jnp.ones((1, 2), dtype=jnp.float32),
        )

        _, exhausted = norm(jnp.ones((1, 1, 4), dtype=jnp.float32), state=state)
        np.testing.assert_array_equal(np.asarray(exhausted.count), [maximum])

        output, unchanged = norm(
            jnp.full((1, 1, 4), jnp.nan, dtype=jnp.float32),
            state=exhausted,
            mask=jnp.zeros((1, 1), dtype=jnp.bool_),
        )
        np.testing.assert_array_equal(np.asarray(output), np.zeros((1, 1, 4)))
        for name in ("count", "mean", "var"):
            np.testing.assert_array_equal(
                np.asarray(getattr(unchanged, name)),
                np.asarray(getattr(exhausted, name)),
            )

    def test_nonfloating_input_is_rejected(self) -> None:
        """The low-level layer honors the documented FP32/BF16 surface."""
        norm = TimestepNorm(4, 2)
        with pytest.raises(TypeError, match="float32 and bfloat16"):
            norm(jnp.ones((1, 1, 4), dtype=jnp.int32))

    def test_divisibility_validation(self) -> None:
        """Test that num_features must be divisible by num_groups.

        :return None: None.
        """
        with pytest.raises(ValueError, match="divisible by"):
            TimestepNorm(63, 8)

    @pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
    def test_epsilon_must_be_finite_and_positive(self, eps: float) -> None:
        """Zero cannot regularize an exactly constant cumulative group."""
        with pytest.raises(ValueError, match="finite and positive"):
            TimestepNorm(4, 2, eps=eps)

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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_contiguous_runs_cover_reused_ids_singletons_and_padding(
        self, random_seed: int
    ) -> None:
        """Every positive contiguous run is independent regardless of raw ID reuse."""
        norm = TimestepNorm(16, 4)
        x = jax.random.normal(jax.random.PRNGKey(random_seed), (1, 9, 16))
        segment_ids = jnp.asarray([[0, 1, 2, 2, 1, 1, 0, 3, 0]], dtype=jnp.int32)

        packed, packed_state = norm(x, segment_ids=segment_ids)

        runs = ((1, 2), (2, 4), (4, 6), (7, 8))
        for start, stop in runs:
            independent, _ = norm(x[:, start:stop])
            np.testing.assert_allclose(
                np.asarray(packed[:, start:stop]),
                np.asarray(independent),
                rtol=2e-5,
                atol=2e-5,
            )
        np.testing.assert_array_equal(
            np.asarray(packed[:, [0, 6, 8]]),
            np.zeros((1, 3, 16)),
        )
        _, final_run_state = norm(x[:, 7:8])
        for name in ("count", "mean", "var"):
            np.testing.assert_allclose(
                np.asarray(getattr(packed_state, name)),
                np.asarray(getattr(final_run_state, name)),
                rtol=2e-5,
                atol=2e-5,
            )

    @pytest.mark.fast
    def test_fully_padded_packed_row_returns_prior(self) -> None:
        """Segment ID zero is an identity and cannot leak nonfinite input."""
        norm = TimestepNorm(8, 2)
        x = jnp.full((1, 7, 8), jnp.nan, dtype=jnp.float32)

        output, state = norm(x, segment_ids=jnp.zeros((1, 7), dtype=jnp.int32))

        np.testing.assert_array_equal(np.asarray(output), np.zeros((1, 7, 8)))
        np.testing.assert_array_equal(np.asarray(state.count), [0])
        np.testing.assert_array_equal(np.asarray(state.mean), np.zeros((1, 2)))
        np.testing.assert_array_equal(np.asarray(state.var), np.ones((1, 2)))

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

    @pytest.mark.fast
    @pytest.mark.parametrize("seq", [1, 31, 32, 33])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    def test_hybrid_matches_sequential_recurrence(
        self, random_seed: int, seq: int, dtype: Any
    ) -> None:
        """Both sides of the FFT/scan threshold match the full recurrence.

        :param int random_seed: Random seed fixture.
        :param int seq: Sequence length around the static dispatch threshold.
        :param Any dtype: Input dtype under test.
        :return None: None.
        """
        key = jax.random.PRNGKey(random_seed)
        k_ema, k_x, k_hr, k_hi = jax.random.split(key, 4)
        ema = ComplexEMA(16, 4, key=k_ema)
        x = jax.random.normal(k_x, (2, 16, seq)).astype(dtype)
        h_init = (
            jax.random.normal(k_hr, (2, 16, 4)) + 1j * jax.random.normal(k_hi, (2, 16, 4))
        ).astype(jnp.complex64) * 0.05
        mask = jnp.ones((2, seq), dtype=jnp.bool_)
        if seq > 1:
            mask = mask.at[:, -2:].set(False)
        masked_x = jnp.where(mask[:, None, :], x, jnp.zeros((), dtype=dtype))
        residual = masked_x.astype(jnp.float32) * ema.omega[None, :, None]

        output_tol = 1e-2 if dtype == jnp.bfloat16 else 2e-4
        for initial in (None, h_init):
            actual_y, actual_h = ema(
                x,
                h_init=initial,
                return_state=True,
                mask=mask,
            )
            reference_y, reference_h = ema._forward_sequential(masked_x, initial)  # noqa: SLF001
            reference_y = (reference_y + residual).astype(dtype)

            assert actual_y.dtype == dtype
            assert actual_h is not None and actual_h.dtype == jnp.complex64
            np.testing.assert_allclose(
                np.asarray(actual_y, dtype=np.float32),
                np.asarray(reference_y, dtype=np.float32),
                rtol=output_tol,
                atol=output_tol,
            )
            np.testing.assert_allclose(
                np.asarray(actual_h),
                np.asarray(reference_h),
                rtol=3e-5,
                atol=3e-6,
            )

            output_only, omitted_state = ema(
                x,
                h_init=initial,
                return_state=False,
                mask=mask,
            )
            assert omitted_state is None
            np.testing.assert_array_equal(np.asarray(output_only), np.asarray(actual_y))

    def test_empty_sequence_preserves_state_contract(self, random_seed: int) -> None:
        """An empty sequence returns empty output and leaves incoming state unchanged."""
        key = jax.random.PRNGKey(random_seed)
        k_ema, k_hr, k_hi = jax.random.split(key, 3)
        ema = ComplexEMA(8, 4, key=k_ema)
        x = jnp.zeros((2, 8, 0), dtype=jnp.bfloat16)
        h_init = (
            jax.random.normal(k_hr, (2, 8, 4)) + 1j * jax.random.normal(k_hi, (2, 8, 4))
        ).astype(jnp.complex64)

        pristine_y, pristine_h = ema(x, return_state=True)
        continuation_y, continuation_h = ema(x, h_init=h_init, return_state=True)
        output_only, omitted_state = ema(x, h_init=h_init, return_state=False)

        assert pristine_y.shape == continuation_y.shape == output_only.shape == x.shape
        assert pristine_y.dtype == continuation_y.dtype == output_only.dtype == x.dtype
        assert pristine_h is not None
        np.testing.assert_array_equal(np.asarray(pristine_h), np.zeros((2, 8, 4), np.complex64))
        np.testing.assert_array_equal(np.asarray(continuation_h), np.asarray(h_init))
        assert omitted_state is None

    def test_nonmultiple_long_block_matches_sequential(self, random_seed: int) -> None:
        """Mapped power blocks pad and slice a nonmultiple length without changing semantics."""
        key = jax.random.PRNGKey(random_seed)
        k_ema, k_x, k_hr, k_hi = jax.random.split(key, 4)
        ema = ComplexEMA(4, 2, key=k_ema)
        seq = 4099
        x = jax.random.normal(k_x, (1, 4, seq)).astype(jnp.bfloat16)
        h_init = (
            jax.random.normal(k_hr, (1, 4, 2)) + 1j * jax.random.normal(k_hi, (1, 4, 2))
        ).astype(jnp.complex64) * 0.05
        mask = jnp.arange(seq)[None, :] < seq - 7
        masked_x = jnp.where(mask[:, None, :], x, jnp.zeros((), dtype=x.dtype))

        actual_y, actual_h = ema(x, h_init=h_init, return_state=True, mask=mask)
        reference_y, reference_h = ema._forward_sequential(masked_x, h_init)  # noqa: SLF001
        residual = masked_x.astype(jnp.float32) * ema.omega[None, :, None]
        reference_y = (reference_y + residual).astype(x.dtype)

        np.testing.assert_allclose(
            np.asarray(actual_y, dtype=np.float32),
            np.asarray(reference_y, dtype=np.float32),
            rtol=1e-2,
            atol=1e-2,
        )
        np.testing.assert_allclose(
            np.asarray(actual_h), np.asarray(reference_h), rtol=3e-5, atol=3e-6
        )

    @pytest.mark.fast
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
        batch, seq = 2, 66
        x = jax.random.normal(k2, (batch, dim, seq))

        # Full sequence with FFT and parallel final-state reduction.
        y_full, h_full = jax_ema(x, return_state=True)

        # Both 33-token chunks select the long-chunk path, including the
        # initial-state FFT bias on the second call.
        y1, h1 = jax_ema(x[:, :, :33], return_state=True)
        y2, h2 = jax_ema(x[:, :, 33:], h_init=h1, return_state=True)
        y_chunked = jnp.concatenate([y1, y2], axis=-1)

        # Should match (within numerical tolerance)
        np.testing.assert_allclose(
            np.array(y_chunked),
            np.array(y_full),
            rtol=1e-4,
            atol=1e-5,
            err_msg="Chunked processing should match full sequence",
        )
        np.testing.assert_allclose(np.asarray(h2), np.asarray(h_full), rtol=2e-5, atol=2e-6)

    @pytest.mark.fast
    def test_long_continuation_gradients_match_sequential(self, random_seed: int) -> None:
        """FFT bias and parallel-state gradients match the recurrent oracle."""
        key = jax.random.PRNGKey(random_seed)
        k_ema, k_x, k_hr, k_hi, k_y, k_sr, k_si = jax.random.split(key, 7)
        ema = ComplexEMA(8, 4, key=k_ema)
        x = jax.random.normal(k_x, (1, 8, 33))
        h_init = (
            jax.random.normal(k_hr, (1, 8, 4)) + 1j * jax.random.normal(k_hi, (1, 8, 4))
        ).astype(jnp.complex64) * 0.05
        output_weight = jax.random.normal(k_y, x.shape)
        state_real_weight = jax.random.normal(k_sr, h_init.shape)
        state_imag_weight = jax.random.normal(k_si, h_init.shape)

        def objective(
            module: ComplexEMA,
            values: jax.Array,
            initial: jax.Array,
            *,
            sequential: bool,
        ) -> jax.Array:
            """Weight both sequence output and returned state."""
            if sequential:
                output, state = module._forward_sequential(values, initial)  # noqa: SLF001
                residual = values.astype(jnp.float32) * module.omega[None, :, None]
                output = (output + residual).astype(values.dtype)
            else:
                output, state = module(values, h_init=initial, return_state=True)
                assert state is not None
            return (
                jnp.sum(output.astype(jnp.float32) * output_weight)
                + jnp.sum(state.real * state_real_weight)
                + jnp.sum(state.imag * state_imag_weight)
            )

        hybrid = jax.value_and_grad(
            lambda module, values, initial: objective(module, values, initial, sequential=False),
            argnums=(0, 1, 2),
        )
        recurrent = jax.value_and_grad(
            lambda module, values, initial: objective(module, values, initial, sequential=True),
            argnums=(0, 1, 2),
        )
        value_hybrid, grads_hybrid = hybrid(ema, x, h_init)
        value_recurrent, grads_recurrent = recurrent(ema, x, h_init)

        np.testing.assert_allclose(
            np.asarray(value_hybrid), np.asarray(value_recurrent), rtol=2e-4, atol=2e-4
        )
        for actual, expected in zip(
            jax.tree.leaves(grads_hybrid), jax.tree.leaves(grads_recurrent), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=8e-4,
                atol=8e-5,
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

    @pytest.mark.fast
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

    @pytest.mark.fast
    def test_associative_matches_sequential_gradients(self, random_seed: int) -> None:
        """Packed implementations agree for full parameter and input gradients."""
        key = jax.random.PRNGKey(random_seed)
        k_ema, k_x, k_weight = jax.random.split(key, 3)
        ema = ComplexEMA(16, 4, key=k_ema)
        x = jax.random.normal(k_x, (2, 16, 20))
        weight = jax.random.normal(k_weight, x.shape)
        segment_ids = jnp.asarray(
            [[1] * 7 + [2] * 8 + [0] * 5, [1] * 5 + [2] * 6 + [1] * 9],
            dtype=jnp.int32,
        )

        def objective(module: ComplexEMA, values: jax.Array, associative: bool) -> jax.Array:
            """Return a weighted packed output objective."""
            output, _ = module(
                values,
                segment_ids=segment_ids,
                use_associative_segment_scan=associative,
            )
            return jnp.sum(output.astype(jnp.float32) * weight)

        value_assoc, grads_assoc = jax.value_and_grad(
            lambda module, values: objective(module, values, True), argnums=(0, 1)
        )(ema, x)
        value_seq, grads_seq = jax.value_and_grad(
            lambda module, values: objective(module, values, False), argnums=(0, 1)
        )(ema, x)

        np.testing.assert_allclose(
            np.asarray(value_assoc), np.asarray(value_seq), rtol=1e-5, atol=2e-5
        )
        for actual, expected in zip(
            jax.tree.leaves(grads_assoc), jax.tree.leaves(grads_seq), strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=2e-4,
                atol=1e-5,
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

    @pytest.mark.fast
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

        ema_bf16 = jax.tree.map(floating_to_bf16, ema_f32)

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

    @pytest.mark.fast
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

        jax_ema_bf16 = jax.tree.map(floating_to_bf16, jax_ema)

        # Generate bf16 input
        batch, seq = 2, 32
        x_bf16 = jax.random.normal(k2, (batch, dim, seq)).astype(jnp.bfloat16)

        # FFT path
        y_fft, _ = jax_ema_bf16(x_bf16, return_state=False)

        # Explicit sequential oracle. ``return_state=True`` no longer implies
        # sequential output for a pristine prefill.
        y_seq, _ = jax_ema_bf16._forward_sequential(x_bf16, None)  # noqa: SLF001
        residual = (
            x_bf16.astype(jnp.float32) * jax_ema_bf16.omega.astype(jnp.float32)[None, :, None]
        )
        y_seq = (y_seq + residual).astype(jnp.bfloat16)

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
