"""Pure-JAX TimestepNorm candidate and prefix-network regression tests."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from megalodon_jax.layers import TimestepNorm
from megalodon_jax.types import NormState
from tools.timestep_norm_candidates import (
    CANDIDATES,
    ResetSummary,
    Summary,
    candidate_a_associative,
    candidate_b_hillis_steele,
    candidate_c_shifted_cumsum,
    chan_merge,
    inclusive_hillis_steele_scan,
    reset_merge,
    serial_stats_only_oracle,
)


def _assert_state_close(actual: NormState, expected: NormState, *, atol: float) -> None:
    np.testing.assert_array_equal(np.asarray(actual.count), np.asarray(expected.count))
    np.testing.assert_allclose(
        np.asarray(actual.mean), np.asarray(expected.mean), atol=atol, rtol=atol
    )
    np.testing.assert_allclose(
        np.asarray(actual.var), np.asarray(expected.var), atol=atol, rtol=atol
    )


def test_chan_merge_uses_m2_and_exact_identities() -> None:
    """Candidate summaries follow the required Chan M2 representation."""
    left = Summary(
        count=jnp.asarray([[[2]]], dtype=jnp.int32),
        mean=jnp.asarray([[[3.0]]], dtype=jnp.float32),
        m2=jnp.asarray([[[4.0]]], dtype=jnp.float32),
    )
    right = Summary(
        count=jnp.asarray([[[3]]], dtype=jnp.int32),
        mean=jnp.asarray([[[7.0]]], dtype=jnp.float32),
        m2=jnp.asarray([[[6.0]]], dtype=jnp.float32),
    )
    identity = Summary(
        count=jnp.zeros((1, 1, 1), dtype=jnp.int32),
        mean=jnp.full((1, 1, 1), -123.0),
        m2=jnp.full((1, 1, 1), 456.0),
    )

    merged = chan_merge(left, right)

    np.testing.assert_array_equal(np.asarray(merged.count), [[[5]]])
    np.testing.assert_allclose(np.asarray(merged.mean), [[[5.4]]], atol=1e-6)
    np.testing.assert_allclose(np.asarray(merged.m2), [[[29.2]]], atol=1e-6)
    for expected, actual in zip(left, chan_merge(identity, left), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    for expected, actual in zip(left, chan_merge(left, identity), strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize("axis", [0, 1, -2])
def test_hillis_steele_matches_associative_scan_for_pytrees(axis: int) -> None:
    """The static network supports pytrees, arbitrary axes, and length seven."""
    shape = (2, 7, 3)
    elements = {
        "a": jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape),
        "b": jnp.ones(shape, dtype=jnp.float32),
    }
    scan_axis = axis
    if axis == 0:
        elements = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), elements)

    expected = jax.lax.associative_scan(
        lambda left, right: jax.tree.map(jnp.add, left, right),
        elements,
        axis=scan_axis,
    )
    actual = inclusive_hillis_steele_scan(
        lambda left, right: jax.tree.map(jnp.add, left, right),
        elements,
        axis=scan_axis,
    )

    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(np.asarray(actual_leaf), np.asarray(expected_leaf))


def test_hillis_steele_preserves_noncommutative_operand_order() -> None:
    """Matrix prefixes prove chronological order rather than commutativity."""
    matrices = jnp.asarray(
        [
            [[1.0, 2.0], [0.0, 1.0]],
            [[2.0, 0.0], [1.0, 1.0]],
            [[1.0, 0.0], [3.0, 2.0]],
            [[0.5, 1.0], [1.0, 0.0]],
            [[2.0, 1.0], [0.0, 1.0]],
        ]
    )

    def operator(left: jax.Array, right: jax.Array) -> jax.Array:
        return jnp.matmul(left, right)

    expected = jax.lax.associative_scan(operator, matrices, axis=0)
    actual = inclusive_hillis_steele_scan(operator, matrices, axis=0)
    reverse_expected = jax.lax.associative_scan(operator, matrices, axis=0, reverse=True)
    reverse_actual = inclusive_hillis_steele_scan(operator, matrices, axis=0, reverse=True)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), atol=1e-6)
    np.testing.assert_allclose(np.asarray(reverse_actual), np.asarray(reverse_expected), atol=1e-6)


def test_hillis_steele_matches_reset_monoid() -> None:
    """The explicit network preserves rightmost-reset contiguous-run semantics."""
    values = jnp.asarray([[[1.0], [2.0], [4.0], [8.0], [16.0], [32.0], [64.0]]])
    elements = ResetSummary(
        summary=Summary(
            count=jnp.ones((1, 7, 1), dtype=jnp.int32),
            mean=values,
            m2=jnp.zeros_like(values),
        ),
        has_reset=jnp.asarray([[[True], [False], [True], [False], [False], [True], [False]]]),
    )

    expected = jax.lax.associative_scan(reset_merge, elements, axis=1)
    actual = inclusive_hillis_steele_scan(reset_merge, elements, axis=1)

    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf), atol=1e-6)


@pytest.mark.parametrize("candidate", list(CANDIDATES.values()), ids=list(CANDIDATES))
@pytest.mark.parametrize("mode", ["plain", "masked", "packed", "continuation"])
def test_candidates_match_compact_serial_oracle(candidate, mode: str) -> None:
    """All candidate scans preserve forward, state, mask, and reset semantics."""
    norm = TimestepNorm(16, 4)
    x = (jax.random.normal(jax.random.PRNGKey(7), (2, 13, 16)) * 3.0 + 20.0).astype(jnp.float32)
    mask = None
    segment_ids = None
    state = None
    if mode == "masked":
        mask = jnp.asarray(
            [
                [True, True, False, True, True, False, True, True, True, False, True, True, False],
                [False, True, True, True, False, True, True, False, True, True, True, False, False],
            ]
        )
        x = x.at[0, 2].set(jnp.nan)
    elif mode == "packed":
        segment_ids = jnp.asarray(
            [
                [0, 1, 2, 2, 1, 1, 0, 3, 3, 4, 4, 4, 0],
                [5, 5, 5, 0, 0, 2, 2, 6, 6, 6, 7, 0, 0],
            ],
            dtype=jnp.int32,
        )
    elif mode == "continuation":
        state = NormState(
            count=jnp.asarray([7, 11], dtype=jnp.int32),
            mean=jnp.asarray(
                [[8.0, 9.0, 10.0, 11.0], [-4.0, -2.0, 0.0, 2.0]],
                dtype=jnp.float32,
            ),
            var=jnp.asarray(
                [[0.25, 0.5, 1.0, 2.0], [3.0, 2.0, 1.0, 0.5]],
                dtype=jnp.float32,
            ),
        )

    expected_output, expected_state = serial_stats_only_oracle(
        norm,
        x,
        state=state,
        mask=mask,
        segment_ids=segment_ids,
    )
    output, result_state = candidate(
        norm,
        x,
        state=state,
        mask=mask,
        segment_ids=segment_ids,
    )

    np.testing.assert_allclose(
        np.asarray(output), np.asarray(expected_output), rtol=3e-5, atol=3e-5
    )
    _assert_state_close(result_state, expected_state, atol=3e-5)
    if mask is not None:
        np.testing.assert_array_equal(np.asarray(output)[~np.asarray(mask)], 0.0)
    if segment_ids is not None:
        np.testing.assert_array_equal(np.asarray(output)[np.asarray(segment_ids) == 0], 0.0)


@pytest.mark.parametrize(
    "candidate",
    [candidate_a_associative, candidate_b_hillis_steele, candidate_c_shifted_cumsum],
)
def test_candidate_gradients_match_serial_oracle(candidate) -> None:
    """Input, affine, and learned-prior gradients agree across associations."""
    norm = TimestepNorm(8, 8, prior_count=2)
    x = (jax.random.normal(jax.random.PRNGKey(19), (1, 9, 8)) * 2.0 + 10.0).astype(jnp.float32)

    def loss(function, module, values):
        output, state = function(module, values)
        return jnp.sum(jnp.sin(output)) + 0.1 * jnp.sum(state.mean) + 0.03 * jnp.sum(state.var)

    expected = eqx.filter_grad(
        lambda module, values: loss(serial_stats_only_oracle, module, values)
    )(norm, x)
    actual = eqx.filter_grad(lambda module, values: loss(candidate, module, values))(norm, x)

    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=5e-4,
            atol=5e-5,
        )


@pytest.mark.parametrize(
    ("name", "candidate", "expect_while"),
    [
        ("candidate_a", candidate_a_associative, False),
        ("candidate_b", candidate_b_hillis_steele, False),
        ("candidate_c", candidate_c_shifted_cumsum, False),
        ("serial_oracle", serial_stats_only_oracle, True),
    ],
)
def test_candidate_stablehlo_sequence_loop_contract(name, candidate, expect_while: bool) -> None:
    """Long-sequence candidates contain no runtime sequence WhileOp."""
    norm = TimestepNorm(32, 4)
    x = jnp.ones((1, 65, 32), dtype=jnp.float32)
    function = eqx.filter_jit(lambda module, values: candidate(module, values))

    stablehlo = str(function.lower(norm, x).lowered.compiler_ir(dialect="stablehlo"))

    assert ("stablehlo.while" in stablehlo) is expect_while, name
    if not expect_while:
        assert "dynamic_update_slice" not in stablehlo


@pytest.mark.parametrize("mode", ["plain", "masked", "packed"])
def test_production_forward_and_backward_have_no_sequence_while(mode: str) -> None:
    """Selected production paths keep sequence work out of runtime WhileOps."""
    norm = TimestepNorm(32, 4)
    x = jnp.ones((1, 65, 32), dtype=jnp.float32)
    mask = None
    segment_ids = None
    if mode == "masked":
        mask = jnp.arange(65)[None] % 5 != 2
    elif mode == "packed":
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


def test_production_matches_selected_shifted_candidate() -> None:
    """The selected plain hot path is exactly the benchmarked Candidate C."""
    norm = TimestepNorm(32, 4)
    x = (jax.random.normal(jax.random.PRNGKey(29), (2, 127, 32)) + 1e4).astype(jnp.float32)

    expected_output, expected_state = candidate_c_shifted_cumsum(norm, x)
    output, state = norm(x)

    np.testing.assert_array_equal(np.asarray(output), np.asarray(expected_output))
    for name in ("count", "mean", "var"):
        np.testing.assert_array_equal(
            np.asarray(getattr(state, name)),
            np.asarray(getattr(expected_state, name)),
        )
