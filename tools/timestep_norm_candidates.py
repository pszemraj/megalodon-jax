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
"""Pure-JAX TimestepNorm candidates for correctness and performance research.

This module keeps experimental prefix algorithms outside the production layer.
All candidates share the released TimestepNorm contract: token-block counts,
FP32 population moments, learned priors, exact-zero masked outputs, packed-run
resets, and plus-one affine storage.

Candidate A uses ``jax.lax.associative_scan`` over Chan summaries. Candidate B
spells out the same prefix as a generic Hillis--Steele network. Candidate C
evaluates a shifted raw-moment identity with one ``lax.cumsum`` on the common
unmasked, unsegmented path. The compact serial Welford implementation is a
separate oracle: only ``(count, mean, M2)`` passes through ``lax.scan``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from megalodon_jax.layers.segments import segment_boundaries, valid_segment_mask
from megalodon_jax.layers.timestep_norm import TimestepNorm
from megalodon_jax.types import NormState

__all__ = [
    "CANDIDATES",
    "ResetSummary",
    "Summary",
    "associative_timestep_norm",
    "candidate_a_associative",
    "candidate_b_hillis_steele",
    "candidate_c_hot_path_supported",
    "candidate_prefix_variance",
    "candidate_c_shifted_cumsum",
    "chan_merge",
    "hillis_steele_timestep_norm",
    "inclusive_hillis_steele_scan",
    "reset_merge",
    "serial_stats_only_oracle",
    "serial_stats_only_timestep_norm",
    "shifted_cumsum_timestep_norm",
]

PyTree = TypeVar("PyTree")


class Summary(NamedTuple):
    """Population-moment summary with an integer token-block count.

    ``m2`` is the sum of squared deviations, not population variance. Counts
    have a trailing singleton dimension so they broadcast across groups.
    """

    count: Int[Array, "... 1"]
    mean: Float[Array, "... groups"]
    m2: Float[Array, "... groups"]


class ResetSummary(NamedTuple):
    """A moment summary annotated with whether its span contains a reset."""

    summary: Summary
    has_reset: Bool[Array, "... 1"]


class _Prepared(NamedTuple):
    """Validated inputs shared by all research candidates."""

    x: Array
    x_grouped: Array
    initial: NormState
    prior: NormState
    valid: Array | None
    boundaries: Array | None
    block_mean: Array
    block_var: Array
    batch: int
    length: int
    dim: int
    groups: int
    group_size: int
    segmented: bool


def chan_merge(left: Summary, right: Summary) -> Summary:
    """Merge population summaries with the identity-aware Chan formula.

    Counts are converted to FP32 *before* multiplication. This avoids integer
    overflow in the cross term while retaining integer counts in the returned
    summary. Empty summaries are exact left and right identities, including
    their otherwise arbitrary mean and M2 payloads.

    :param Summary left: Summary for the earlier span.
    :param Summary right: Summary for the later span.
    :return Summary: Summary for the concatenated span.
    """
    left_count_f = left.count.astype(jnp.float32)
    right_count_f = right.count.astype(jnp.float32)
    count = left.count + right.count
    count_f = left_count_f + right_count_f
    denominator = jnp.maximum(count_f, 1.0)
    delta = right.mean - left.mean
    merged_mean = left.mean + delta * (right_count_f / denominator)
    merged_m2 = (
        left.m2 + right.m2 + jnp.square(delta) * (left_count_f * right_count_f / denominator)
    )

    left_empty = left.count == 0
    right_empty = right.count == 0
    # Check the right identity first so merging two empty summaries preserves
    # the left payload exactly (notably an incoming zero-count state's mean).
    mean = jnp.where(right_empty, left.mean, jnp.where(left_empty, right.mean, merged_mean))
    m2 = jnp.where(right_empty, left.m2, jnp.where(left_empty, right.m2, merged_m2))
    return Summary(count=count, mean=mean, m2=m2)


def reset_merge(left: ResetSummary, right: ResetSummary) -> ResetSummary:
    """Merge summaries while discarding history before the rightmost reset.

    The operator preserves chronological operand order and is associative when
    ``chan_merge`` is associative. ``right.summary`` already represents the
    suffix beginning at the last reset in the right span.

    :param ResetSummary left: Earlier reset-aware span.
    :param ResetSummary right: Later reset-aware span.
    :return ResetSummary: Reset-aware summary for the joined span.
    """
    merged = chan_merge(left.summary, right.summary)
    summary = _select_summary(right.has_reset, right.summary, merged)
    return ResetSummary(summary=summary, has_reset=left.has_reset | right.has_reset)


def inclusive_hillis_steele_scan(
    operator: Callable[[PyTree, PyTree], PyTree],
    elements: PyTree,
    *,
    axis: int = 0,
    reverse: bool = False,
) -> PyTree:
    """Return an inclusive Hillis--Steele scan over an arbitrary pytree.

    Every array leaf must have the same scan-axis length. Leaf ranks may differ,
    and negative axes are resolved independently for each leaf. The iterative
    offsets ``1, 2, 4, ...`` support non-power-of-two lengths. Operand order is
    preserved for non-commutative associative operators; ``reverse=True``
    returns ordered inclusive suffixes.

    :param operator: Associative binary operation whose operands and result
        share the pytree structure of ``elements``.
    :param elements: Pytree of arrays to scan.
    :param int axis: Scan axis, including negative indexing.
    :param bool reverse: Compute an ordered suffix scan instead of a prefix.
    :raises ValueError: If leaves are absent or have incompatible axes/lengths.
    :raises TypeError: If a leaf is not array-like.
    :return PyTree: Pytree containing inclusive prefixes or suffixes.
    """
    leaves = jax.tree.leaves(elements)
    if not leaves:
        raise ValueError("inclusive_hillis_steele_scan requires at least one array leaf")

    def leaf_axis(leaf: Any) -> int:
        if not hasattr(leaf, "ndim") or not hasattr(leaf, "shape"):
            raise TypeError("inclusive_hillis_steele_scan accepts only array leaves")
        resolved = axis if axis >= 0 else leaf.ndim + axis
        if resolved < 0 or resolved >= leaf.ndim:
            raise ValueError(f"axis {axis} is invalid for leaf shape {leaf.shape}")
        return resolved

    axes = [leaf_axis(leaf) for leaf in leaves]
    length = int(leaves[0].shape[axes[0]])
    if any(int(leaf.shape[leaf_axis_]) != length for leaf, leaf_axis_ in zip(leaves, axes)):
        raise ValueError("all pytree leaves must have the same scan-axis length")
    if length <= 1:
        return elements

    def flip(tree: PyTree) -> PyTree:
        return jax.tree.map(lambda leaf: jnp.flip(leaf, axis=leaf_axis(leaf)), tree)

    result = flip(elements) if reverse else elements
    combine = operator
    offset = 1
    while offset < length:
        left = jax.tree.map(
            lambda leaf: jax.lax.slice_in_dim(
                leaf,
                0,
                length - offset,
                axis=leaf_axis(leaf),
            ),
            result,
        )
        right = jax.tree.map(
            lambda leaf: jax.lax.slice_in_dim(
                leaf,
                offset,
                length,
                axis=leaf_axis(leaf),
            ),
            result,
        )
        merged = combine(left, right)
        leading = jax.tree.map(
            lambda leaf: jax.lax.slice_in_dim(
                leaf,
                0,
                offset,
                axis=leaf_axis(leaf),
            ),
            result,
        )
        result = jax.tree.map(
            lambda prefix, suffix: jnp.concatenate(
                (prefix, suffix),
                axis=leaf_axis(prefix),
            ),
            leading,
            merged,
        )
        offset *= 2
    return flip(result) if reverse else result


def associative_timestep_norm(
    norm: TimestepNorm,
    x: Float[Array, "batch seq dim"],
    state: NormState | None = None,
    mask: Bool[Array, "batch seq"] | None = None,
    segment_ids: Int[Array, "batch seq"] | None = None,
) -> tuple[Float[Array, "batch seq dim"], NormState]:
    """Evaluate Candidate A with ``jax.lax.associative_scan``.

    Token summaries are scanned without prepending the incoming state. The
    initial state or per-run prior is Chan-merged into every completed prefix
    afterward, which keeps the scan network at the original sequence length.
    """
    return _parallel_timestep_norm(
        norm,
        x,
        state,
        mask,
        segment_ids,
        scanner=lambda operator, elements: jax.lax.associative_scan(
            operator,
            elements,
            axis=1,
        ),
    )


def hillis_steele_timestep_norm(
    norm: TimestepNorm,
    x: Float[Array, "batch seq dim"],
    state: NormState | None = None,
    mask: Bool[Array, "batch seq"] | None = None,
    segment_ids: Int[Array, "batch seq"] | None = None,
) -> tuple[Float[Array, "batch seq dim"], NormState]:
    """Evaluate Candidate B with the explicit Hillis--Steele scan network."""
    return _parallel_timestep_norm(
        norm,
        x,
        state,
        mask,
        segment_ids,
        scanner=lambda operator, elements: inclusive_hillis_steele_scan(
            operator,
            elements,
            axis=1,
        ),
    )


def _parallel_timestep_norm(
    norm: TimestepNorm,
    x: Array,
    state: NormState | None,
    mask: Array | None,
    segment_ids: Array | None,
    *,
    scanner: Callable[[Callable[[Any, Any], Any], Any], Any],
) -> tuple[Array, NormState]:
    """Evaluate a parallel Chan-prefix candidate with a supplied scanner."""
    prepared = _prepare(norm, x, state, mask, segment_ids)
    if prepared.length == 0:
        return x, prepared.initial

    prefix, final, fallback = _parallel_prefix(prepared, scanner=scanner)
    output = _normalize(norm, prepared, prefix)
    return output, _summary_to_state(final, fallback)


def _parallel_prefix(
    prepared: _Prepared,
    *,
    scanner: Callable[[Callable[[Any, Any], Any], Any], Any],
) -> tuple[Summary, Summary, NormState]:
    """Return prefix/final summaries for an exact parallel candidate."""

    tokens = _token_summaries(prepared)
    if not prepared.segmented:
        token_prefix = scanner(chan_merge, tokens)
        initial = _broadcast_summary(_state_summary(prepared.initial), tokens)
        prefix = chan_merge(initial, token_prefix)
        final = _slice_summary(prefix, prepared.length - 1, prepared.length, axis=1)
    else:
        assert prepared.boundaries is not None
        assert prepared.valid is not None
        boundary = prepared.boundaries[..., None]
        elements = ResetSummary(
            summary=tokens,
            has_reset=boundary,
        )
        token_prefix = scanner(reset_merge, elements).summary
        prior = _broadcast_summary(_state_summary(prepared.prior), tokens)
        prefix = chan_merge(prior, token_prefix)
        final = _last_valid_summary(prefix, prepared.valid, _state_summary(prepared.prior))

    fallback = prepared.prior if prepared.segmented else prepared.initial
    return prefix, final, fallback


def shifted_cumsum_timestep_norm(
    norm: TimestepNorm,
    x: Float[Array, "batch seq dim"],
    state: NormState | None = None,
    mask: Bool[Array, "batch seq"] | None = None,
    segment_ids: Int[Array, "batch seq"] | None = None,
) -> tuple[Float[Array, "batch seq dim"], NormState]:
    """Evaluate Candidate C: shifted moments with one ``lax.cumsum``.

    The anchor is the incoming mean when history exists, otherwise the first
    valid token-block mean. ``stop_gradient`` makes the algebraic reference
    point a numerical device rather than a gradient path. Shifted sum and
    shifted square-sum are packed into one FP32 array and accumulated by a
    single ``lax.cumsum`` invocation; exact int32 prefix counts come from the
    static unmasked token offsets.

    This hot path is specialized for ``mask is None`` and
    ``segment_ids is None``. Other calls delegate to exact Candidate A; callers
    benchmarking Candidate C should use ``candidate_c_hot_path_supported`` to
    distinguish the hot path from that semantic fallback.
    """
    if not candidate_c_hot_path_supported(mask, segment_ids):
        return associative_timestep_norm(norm, x, state, mask, segment_ids)

    prepared = _prepare(norm, x, state, mask, segment_ids)
    if prepared.length == 0:
        return x, prepared.initial
    prefix, variance = _shifted_prefix(prepared)
    output = _normalize_from_moments(norm, prepared, prefix.mean, variance)
    return output, NormState(
        count=prefix.count[:, -1, 0],
        mean=prefix.mean[:, -1],
        var=variance[:, -1],
    )


def _shifted_prefix(prepared: _Prepared) -> tuple[Summary, Array]:
    """Return Candidate C prefixes and their direct population variance."""
    first_mean = prepared.block_mean[:, 0]
    anchor = jnp.where(
        (prepared.initial.count > 0)[:, None],
        prepared.initial.mean,
        first_mean,
    )
    anchor = jax.lax.stop_gradient(anchor.astype(jnp.float32))

    initial_count_f = prepared.initial.count.astype(jnp.float32)[:, None]
    initial_delta = prepared.initial.mean - anchor
    initial_m2 = prepared.initial.var * initial_count_f
    initial_shifted_sum = initial_delta * initial_count_f
    initial_shifted_square = initial_m2 + jnp.square(initial_delta) * initial_count_f
    token_delta = prepared.block_mean - anchor[:, None]
    token_shifted_sum = token_delta
    token_shifted_square = prepared.block_var + jnp.square(token_delta)
    token_rows = jnp.concatenate(
        (token_shifted_sum, token_shifted_square),
        axis=-1,
    )

    packed_prefix = jax.lax.cumsum(token_rows, axis=1)
    groups = prepared.groups
    shifted_sum = packed_prefix[..., :groups] + initial_shifted_sum[:, None]
    shifted_square = packed_prefix[..., groups:] + initial_shifted_square[:, None]
    offsets = jnp.arange(1, prepared.length + 1, dtype=jnp.int32)[None]
    count = prepared.initial.count[:, None] + offsets
    count_f = count.astype(jnp.float32)[..., None]
    offset = shifted_sum / count_f
    mean = anchor[:, None] + offset
    variance = shifted_square / count_f - jnp.square(offset)
    prefix = Summary(
        count=count[..., None],
        mean=mean,
        m2=variance * count_f,
    )
    return prefix, variance


def candidate_c_hot_path_supported(
    mask: Array | None,
    segment_ids: Array | None,
) -> bool:
    """Return whether Candidate C will execute its one-cumsum hot path."""
    return mask is None and segment_ids is None


def serial_stats_only_timestep_norm(
    norm: TimestepNorm,
    x: Float[Array, "batch seq dim"],
    state: NormState | None = None,
    mask: Bool[Array, "batch seq"] | None = None,
    segment_ids: Int[Array, "batch seq"] | None = None,
) -> tuple[Float[Array, "batch seq dim"], NormState]:
    """Evaluate the compact serial Welford oracle.

    Only ``Summary`` and the last-valid state traverse ``lax.scan``. Token
    moments and D-wide normalization/affine work remain fully vectorized.
    """
    prepared = _prepare(norm, x, state, mask, segment_ids)
    if prepared.length == 0:
        return x, prepared.initial

    prefix, final, fallback = _serial_prefix(prepared)
    output = _normalize(norm, prepared, prefix)
    return output, _summary_to_state(final, fallback)


def _serial_prefix(prepared: _Prepared) -> tuple[Summary, Summary, NormState]:
    """Return prefixes from the compact statistics-only serial oracle."""

    tokens = _token_summaries(prepared)
    tokens_t = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), tokens)
    boundaries = (
        jnp.zeros((prepared.batch, prepared.length), dtype=jnp.bool_)
        if prepared.boundaries is None
        else prepared.boundaries
    )
    boundaries_t = jnp.swapaxes(boundaries[..., None], 0, 1)
    initial = _compact_state_summary(prepared.initial)
    prior = _compact_state_summary(prepared.prior)

    def step(
        carry: tuple[Summary, Summary],
        inputs: tuple[Summary, Array],
    ) -> tuple[tuple[Summary, Summary], Summary]:
        current, last_valid = carry
        token, boundary = inputs
        current = _select_summary(boundary, prior, current)
        current = chan_merge(current, token)
        last_valid = _select_summary(token.count > 0, current, last_valid)
        return (current, last_valid), current

    (_, final), prefix_t = jax.lax.scan(
        step,
        (initial, initial),
        (tokens_t, boundaries_t),
    )
    prefix = jax.tree.map(lambda leaf: jnp.swapaxes(leaf, 0, 1), prefix_t)
    final = Summary(
        count=final.count[:, None],
        mean=final.mean[:, None],
        m2=final.m2[:, None],
    )
    fallback = prepared.prior if prepared.segmented else prepared.initial
    return prefix, final, fallback


def candidate_prefix_variance(
    candidate_name: str,
    norm: TimestepNorm,
    x: Float[Array, "batch seq dim"],
    state: NormState | None = None,
    mask: Bool[Array, "batch seq"] | None = None,
    segment_ids: Int[Array, "batch seq"] | None = None,
) -> Float[Array, "batch seq groups"]:
    """Return every candidate prefix variance for numerical diagnostics.

    Candidate C reports the direct shifted-moment variance rather than
    reconstructing it from its diagnostic M2 payload. Masked and packed calls
    use Candidate A, matching Candidate C's documented semantic fallback.

    :param str candidate_name: Exact key from ``CANDIDATES``.
    :param TimestepNorm norm: Normalization module.
    :param Array x: Input activation sequence.
    :param NormState | None state: Optional continuation state.
    :param Array | None mask: Valid-token mask.
    :param Array | None segment_ids: Packed segment identifiers.
    :raises ValueError: If ``candidate_name`` is unknown.
    :return Array: FP32 population variance at every token prefix.
    """
    prepared = _prepare(norm, x, state, mask, segment_ids)
    if prepared.length == 0:
        return jnp.zeros(
            (prepared.batch, 0, prepared.groups),
            dtype=jnp.float32,
        )

    def associative_scanner(operator: Callable[[Any, Any], Any], elements: Any) -> Any:
        return jax.lax.associative_scan(operator, elements, axis=1)

    def hillis_scanner(operator: Callable[[Any, Any], Any], elements: Any) -> Any:
        return inclusive_hillis_steele_scan(operator, elements, axis=1)

    if candidate_name == "candidate_a_associative":
        prefix, _, fallback = _parallel_prefix(prepared, scanner=associative_scanner)
        return _summary_variance(prefix, fallback)
    if candidate_name == "candidate_b_hillis_steele":
        prefix, _, fallback = _parallel_prefix(prepared, scanner=hillis_scanner)
        return _summary_variance(prefix, fallback)
    if candidate_name in ("candidate_c_shifted_cumsum", "production"):
        if candidate_c_hot_path_supported(mask, segment_ids):
            _, variance = _shifted_prefix(prepared)
            return variance
        prefix, _, fallback = _parallel_prefix(prepared, scanner=associative_scanner)
        return _summary_variance(prefix, fallback)
    if candidate_name == "serial_stats_only_oracle":
        prefix, _, fallback = _serial_prefix(prepared)
        return _summary_variance(prefix, fallback)
    raise ValueError(f"unknown TimestepNorm candidate: {candidate_name}")


# Exact public names used by the candidate verifier and benchmark.
candidate_a_associative = associative_timestep_norm
candidate_b_hillis_steele = hillis_steele_timestep_norm
candidate_c_shifted_cumsum = shifted_cumsum_timestep_norm
serial_stats_only_oracle = serial_stats_only_timestep_norm

CANDIDATES: dict[
    str,
    Callable[
        [TimestepNorm, Array, NormState | None, Array | None, Array | None],
        tuple[Array, NormState],
    ],
] = {
    "candidate_a_associative": candidate_a_associative,
    "candidate_b_hillis_steele": candidate_b_hillis_steele,
    "candidate_c_shifted_cumsum": candidate_c_shifted_cumsum,
    "serial_stats_only_oracle": serial_stats_only_oracle,
}


def _prepare(
    norm: TimestepNorm,
    x: Array,
    state: NormState | None,
    mask: Array | None,
    segment_ids: Array | None,
) -> _Prepared:
    """Validate public inputs and build vectorized token-block moments."""
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 input, got shape {x.shape}")
    batch, length, dim = x.shape
    if dim != norm.num_features:
        raise ValueError(f"expected {norm.num_features} features, got {dim}")
    if x.dtype not in (jnp.float32, jnp.bfloat16):
        raise TypeError("TimestepNorm candidates support only float32 and bfloat16")
    if segment_ids is not None and state is not None:
        raise ValueError("segment_ids cannot be combined with an incoming NormState")

    prior = norm._prior_state(batch)  # noqa: SLF001 - research helper mirrors the layer.
    if state is None:
        initial = prior
    else:
        expected = (batch, norm.num_groups)
        if state.count.shape != (batch,):
            raise ValueError(f"state.count must have shape {(batch,)}, got {state.count.shape}")
        if state.mean.shape != expected or state.var.shape != expected:
            raise ValueError(
                f"state mean/var must have shape {expected}, "
                f"got {state.mean.shape} and {state.var.shape}"
            )
        initial = NormState(
            count=state.count.astype(jnp.int32),
            mean=state.mean.astype(jnp.float32),
            var=state.var.astype(jnp.float32),
        )

    valid = None
    if mask is not None:
        if mask.shape != (batch, length):
            raise ValueError(f"mask must have shape {(batch, length)}, got {mask.shape}")
        valid = mask.astype(jnp.bool_)

    segmented = segment_ids is not None
    boundaries = None
    if segment_ids is not None:
        if segment_ids.shape != (batch, length):
            raise ValueError(
                f"segment_ids must have shape {(batch, length)}, got {segment_ids.shape}"
            )
        segment_valid = valid_segment_mask(segment_ids)
        valid = segment_valid if valid is None else valid & segment_valid
        boundaries = segment_boundaries(segment_ids)

    if state is None:
        if norm.prior_count > jnp.iinfo(jnp.int32).max - length:
            raise ValueError("TimestepNorm int32 count would overflow")
        initial_count = initial.count
    else:
        increment = (
            jnp.asarray(length, dtype=jnp.int32)
            if valid is None
            else valid.astype(jnp.int32).sum(axis=1)
        )
        initial_count = eqx.error_if(
            initial.count,
            jnp.any((initial.count < 0) | (initial.count > jnp.iinfo(jnp.int32).max - increment)),
            "TimestepNorm count must be non-negative and must not overflow int32",
        )
    initial = NormState(initial_count, initial.mean, initial.var)

    groups = norm.num_groups
    group_size = norm.group_size
    x_grouped = x.astype(jnp.float32).reshape(batch, length, groups, group_size)
    moment_groups = (
        x_grouped if valid is None else jnp.where(valid[..., None, None], x_grouped, 0.0)
    )
    block_mean = jnp.mean(moment_groups, axis=-1)
    block_var = jnp.mean(jnp.square(moment_groups - block_mean[..., None]), axis=-1)
    return _Prepared(
        x=x,
        x_grouped=moment_groups,
        initial=initial,
        prior=prior,
        valid=valid,
        boundaries=boundaries,
        block_mean=block_mean,
        block_var=block_var,
        batch=batch,
        length=length,
        dim=dim,
        groups=groups,
        group_size=group_size,
        segmented=segmented,
    )


def _state_summary(state: NormState) -> Summary:
    """Convert a batch state to a singleton-length M2 summary."""
    count = state.count[:, None, None].astype(jnp.int32)
    count_f = count.astype(jnp.float32)
    return Summary(
        count=count,
        mean=state.mean[:, None].astype(jnp.float32),
        m2=state.var[:, None].astype(jnp.float32) * count_f,
    )


def _compact_state_summary(state: NormState) -> Summary:
    """Convert a batch state to the compact carry shape used by ``lax.scan``."""
    count = state.count[:, None].astype(jnp.int32)
    count_f = count.astype(jnp.float32)
    return Summary(
        count=count,
        mean=state.mean.astype(jnp.float32),
        m2=state.var.astype(jnp.float32) * count_f,
    )


def _token_summaries(prepared: _Prepared) -> Summary:
    """Build identity-safe per-token summaries from vectorized moments."""
    if prepared.valid is None:
        return Summary(
            count=jnp.ones(
                (prepared.batch, prepared.length, 1),
                dtype=jnp.int32,
            ),
            mean=prepared.block_mean,
            m2=prepared.block_var,
        )
    valid = prepared.valid[..., None]
    return Summary(
        count=valid.astype(jnp.int32),
        mean=jnp.where(valid, prepared.block_mean, 0.0),
        m2=jnp.where(valid, prepared.block_var, 0.0),
    )


def _select_summary(predicate: Array, if_true: Summary, if_false: Summary) -> Summary:
    """Select whole summaries with a trailing-singleton predicate."""
    return Summary(
        count=jnp.where(predicate, if_true.count, if_false.count),
        mean=jnp.where(predicate, if_true.mean, if_false.mean),
        m2=jnp.where(predicate, if_true.m2, if_false.m2),
    )


def _broadcast_summary(summary: Summary, template: Summary) -> Summary:
    """Broadcast a singleton-length summary to a token-summary shape."""
    return Summary(
        count=jnp.broadcast_to(summary.count, template.count.shape),
        mean=jnp.broadcast_to(summary.mean, template.mean.shape),
        m2=jnp.broadcast_to(summary.m2, template.m2.shape),
    )


def _slice_summary(summary: Summary, start: int, limit: int, *, axis: int) -> Summary:
    """Slice every summary leaf along one axis."""
    return Summary(*(jax.lax.slice_in_dim(leaf, start, limit, axis=axis) for leaf in summary))


def _last_valid_summary(prefix: Summary, valid: Array, fallback: Summary) -> Summary:
    """Gather each row's last valid prefix or a singleton fallback."""
    length = valid.shape[1]
    positions = jnp.arange(length, dtype=jnp.int32)[None]
    last_valid = jnp.max(jnp.where(valid, positions, -1), axis=1)
    index = jnp.maximum(last_valid, 0)

    def gather(leaf: Array) -> Array:
        index_shape = (leaf.shape[0], 1) + (1,) * (leaf.ndim - 2)
        indices = index.reshape(index_shape)
        return jnp.take_along_axis(leaf, indices, axis=1)

    gathered = jax.tree.map(gather, prefix)
    return _select_summary((last_valid >= 0)[:, None, None], gathered, fallback)


def _summary_variance(summary: Summary, fallback: NormState) -> Array:
    """Convert M2 to population variance with an empty-state fallback."""
    count_f = summary.count.astype(jnp.float32)
    variance = summary.m2 / jnp.maximum(count_f, 1.0)
    fallback_var = fallback.var[:, None]
    return jnp.where(summary.count > 0, variance, fallback_var)


def _summary_to_state(summary: Summary, fallback: NormState) -> NormState:
    """Convert a singleton-length summary to the public continuation state."""
    return NormState(
        count=summary.count[:, 0, 0],
        mean=summary.mean[:, 0],
        var=_summary_variance(summary, fallback)[:, 0],
    )


def _normalize(norm: TimestepNorm, prepared: _Prepared, prefix: Summary) -> Array:
    """Apply vectorized normalization, plus-one affine, and exact-zero masking."""
    fallback = prepared.prior if prepared.segmented else prepared.initial
    variance = _summary_variance(prefix, fallback)
    return _normalize_from_moments(norm, prepared, prefix.mean, variance)


def _normalize_from_moments(
    norm: TimestepNorm,
    prepared: _Prepared,
    mean: Array,
    variance: Array,
) -> Array:
    """Apply vectorized normalization from explicit prefix mean/variance arrays."""
    centered = prepared.x_grouped - mean[..., None]
    normalized = centered * jax.lax.rsqrt(variance[..., None] + norm.eps)
    scale = (norm.weight + 1.0).reshape(prepared.groups, prepared.group_size)
    bias = norm.bias.reshape(prepared.groups, prepared.group_size)
    output = normalized * scale + bias
    if prepared.valid is not None:
        output = jnp.where(prepared.valid[..., None, None], output, 0.0)
    return output.reshape(prepared.batch, prepared.length, prepared.dim).astype(prepared.x.dtype)
