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
"""Source-compatible causal TimestepNorm for Megalodon JAX."""

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from megalodon_jax.layers.segments import segment_boundaries, valid_segment_mask
from megalodon_jax.types import NormState


def _block_moments(x: Array) -> tuple[Array, Array]:
    """Return stable population moments over the final feature axis.

    :param Array x: Grouped input values.
    :return tuple[Array, Array]: Population mean and variance for each group.
    """
    mean = jnp.mean(x, axis=-1)
    var = jnp.mean(jnp.square(x - mean[..., None]), axis=-1)
    return mean, var


class _MomentSummary(NamedTuple):
    """Equal-sized token-block moments with an unnormalized second moment."""

    count: Array
    mean: Array
    m2: Array


class _SegmentedSummary(NamedTuple):
    """Moment summary plus whether its interval contains a reset."""

    count: Array
    mean: Array
    m2: Array
    has_reset: Array


def _merge_m2(left: _MomentSummary, right: _MomentSummary) -> _MomentSummary:
    """Associatively merge Chan--Golub--LeVeque block summaries.

    :param _MomentSummary left: Summary for the earlier interval.
    :param _MomentSummary right: Summary for the later interval.
    :return _MomentSummary: Summary for the concatenated interval.
    """
    count_a, mean_a, m2_a = left
    count_b, mean_b, m2_b = right
    count = count_a + count_b
    count_a_f = count_a.astype(jnp.float32)
    count_b_f = count_b.astype(jnp.float32)
    count_f = jnp.maximum(count.astype(jnp.float32), 1.0)
    delta = mean_b - mean_a
    merged_mean = mean_a + delta * (count_b_f / count_f)
    merged_m2 = m2_a + m2_b + jnp.square(delta) * (count_a_f * count_b_f / count_f)

    left_empty = count_a == 0
    right_empty = count_b == 0
    mean = jnp.where(right_empty, mean_a, jnp.where(left_empty, mean_b, merged_mean))
    m2 = jnp.where(right_empty, m2_a, jnp.where(left_empty, m2_b, merged_m2))
    return _MomentSummary(count=count, mean=mean, m2=m2)


def _merge_segmented_m2(
    left: _SegmentedSummary,
    right: _SegmentedSummary,
) -> _SegmentedSummary:
    """Merge summaries while discarding everything before a right-hand reset.

    :param _SegmentedSummary left: Summary for the earlier interval.
    :param _SegmentedSummary right: Summary for the later interval.
    :return _SegmentedSummary: Reset-aware summary for the concatenated interval.
    """
    merged = _merge_m2(
        _MomentSummary(left.count, left.mean, left.m2),
        _MomentSummary(right.count, right.mean, right.m2),
    )
    choose_right = right.has_reset
    return _SegmentedSummary(
        count=jnp.where(choose_right, right.count, merged.count),
        mean=jnp.where(choose_right, right.mean, merged.mean),
        m2=jnp.where(choose_right, right.m2, merged.m2),
        has_reset=left.has_reset | right.has_reset,
    )


def _shifted_cumsum_prefix(
    block_mean: Array,
    block_var: Array,
    initial: NormState,
) -> tuple[Array, Array, Array]:
    """Compute an unmasked prefix from one shifted first/second-moment cumsum.

    "Shifted" refers to subtracting a numerical anchor from the moments; it
    does not shift values along the sequence axis.

    :param Array block_mean: Per-token population means by group.
    :param Array block_var: Per-token population variances by group.
    :param NormState initial: Incoming causal normalization state.
    :return tuple[Array, Array, Array]: Prefix counts, means, and population variances.
    """
    _, length, groups = block_mean.shape
    anchor = jnp.where(initial.count[:, None] > 0, initial.mean, block_mean[:, 0])
    anchor = jax.lax.stop_gradient(anchor)
    shifted_mean = block_mean - anchor[:, None, :]
    shifted_second = block_var + jnp.square(shifted_mean)
    prefix_moments = jax.lax.cumsum(
        jnp.concatenate((shifted_mean, shifted_second), axis=-1),
        axis=1,
    )
    prefix_first = prefix_moments[..., :groups]
    prefix_second = prefix_moments[..., groups:]

    initial_count_f = initial.count.astype(jnp.float32)[:, None]
    initial_offset = initial.mean - anchor
    initial_first = initial_count_f * initial_offset
    initial_second = initial_count_f * (initial.var + jnp.square(initial_offset))

    steps = jnp.arange(1, length + 1, dtype=jnp.int32)[None, :, None]
    count = initial.count[:, None, None] + steps
    count_f = count.astype(jnp.float32)
    offset = (initial_first[:, None, :] + prefix_first) / count_f
    mean = anchor[:, None, :] + offset
    var = (initial_second[:, None, :] + prefix_second) / count_f - jnp.square(offset)
    return count, mean, var


class TimestepNorm(eqx.Module):
    """Normalize each group over all valid scalars observed up to a timestep.

    State count follows the released implementation and counts equal-sized valid
    token blocks. Each block contributes its population mean and variance over
    the features in a group, which is algebraically equivalent to counting every
    scalar in the paper equation.

    Stored affine scale uses the released plus-one parameterization:
    effective_scale = weight + 1.
    """

    num_features: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    group_size: int = eqx.field(static=True)
    prior_count: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    prior_mean: Float[Array, "groups"] | None
    prior_logv: Float[Array, "groups"] | None
    weight: Float[Array, "dim"]
    bias: Float[Array, "dim"]

    def __init__(
        self,
        num_features: int,
        num_groups: int | None = None,
        prior_count: int = 0,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        """Initialize TimestepNorm.

        :param int num_features: Total feature width.
        :param int | None num_groups: Number of statistic groups. None or
            num_features selects featurewise TimestepNorm.
        :param int prior_count: Number of learned-prior observations.
        :param float eps: Numerical epsilon added to population variance.
        :param bool affine: Compatibility argument; must be true because released
            TimestepNorm is always affine.
        :raises ValueError: If dimensions, prior count, epsilon, or affine mode are invalid.
        """
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if prior_count < 0:
            raise ValueError(f"prior_count must be non-negative, got {prior_count}")
        if prior_count > jnp.iinfo(jnp.int32).max:
            raise ValueError("prior_count exceeds the supported int32 state range")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}")
        if not affine:
            raise ValueError("TimestepNorm is always affine in the released architecture")

        featurewise = num_groups is None or num_groups == num_features
        resolved_groups = num_features if featurewise else num_groups
        assert resolved_groups is not None
        if featurewise and prior_count <= 1:
            raise ValueError("featurewise TimestepNorm requires prior_count > 1")
        if num_features % resolved_groups != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by num_groups ({resolved_groups})"
            )

        self.num_features = num_features
        self.num_groups = resolved_groups
        self.group_size = num_features // resolved_groups
        self.prior_count = prior_count
        self.eps = eps
        if prior_count > 0:
            self.prior_mean = jnp.zeros(resolved_groups, dtype=jnp.float32)
            self.prior_logv = jnp.zeros(resolved_groups, dtype=jnp.float32)
        else:
            self.prior_mean = None
            self.prior_logv = None
        self.weight = jnp.zeros(num_features, dtype=jnp.float32)
        self.bias = jnp.zeros(num_features, dtype=jnp.float32)

    def _prior_state(self, batch_size: int) -> NormState:
        """Create the source-compatible initial state for a batch.

        :param int batch_size: Number of rows in the batch.
        :return NormState: Broadcast learned prior or default zero-history state.
        """
        count = jnp.full((batch_size,), self.prior_count, dtype=jnp.int32)
        if self.prior_mean is None:
            mean = jnp.zeros((batch_size, self.num_groups), dtype=jnp.float32)
            var = jnp.ones((batch_size, self.num_groups), dtype=jnp.float32)
        else:
            assert self.prior_logv is not None
            mean = jnp.broadcast_to(
                self.prior_mean.astype(jnp.float32), (batch_size, self.num_groups)
            )
            var = jnp.broadcast_to(
                jnp.exp(self.prior_logv.astype(jnp.float32)),
                (batch_size, self.num_groups),
            )
        return NormState(count=count, mean=mean, var=var)

    def __call__(
        self,
        x: Float[Array, "batch seq dim"],
        state: NormState | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
        *,
        _state_validated: bool = False,
    ) -> tuple[Float[Array, "batch seq dim"], NormState]:
        """Normalize a sequence and return its final causal statistic state.

        Public masks use True for valid tokens. Segment id zero is padding;
        every contiguous nonzero segment is normalized as an independent sequence.
        Masked outputs are exactly zero after affine transformation.

        :param Float[Array, "batch seq dim"] x: Input activations.
        :param NormState | None state: Optional continuation state.
        :param Bool[Array, "batch seq"] | None mask: Valid-token mask.
        :param Int[Array, "batch seq"] | None segment_ids: Packed-sequence ids.
        :param bool _state_validated: Internal model-path signal that cache counts were checked.
        :raises ValueError: If shapes are invalid or packed resets are combined with state.
        :raises TypeError: If float16 input is provided.
        :return tuple: Normalized output and final population-statistic state.
        """
        if x.ndim != 3:
            raise ValueError(f"TimestepNorm expects rank-3 input, got shape {x.shape}")
        batch_size, length, dim = x.shape
        if dim != self.num_features:
            raise ValueError(f"TimestepNorm expected {self.num_features} features, got {dim}")
        if x.dtype not in (jnp.float32, jnp.bfloat16):
            raise TypeError("TimestepNorm supports only float32 and bfloat16")
        if segment_ids is not None and state is not None:
            raise ValueError("segment_ids cannot be combined with an incoming NormState")

        prior = self._prior_state(batch_size)
        if state is None:
            initial = prior
        else:
            expected_mean_shape = (batch_size, self.num_groups)
            if state.count.shape != (batch_size,):
                raise ValueError(
                    f"NormState.count must have shape {(batch_size,)}, got {state.count.shape}"
                )
            if state.mean.shape != expected_mean_shape or state.var.shape != expected_mean_shape:
                raise ValueError(
                    f"NormState mean/var must have shape {expected_mean_shape}, "
                    f"got {state.mean.shape} and {state.var.shape}"
                )
            initial = NormState(
                count=state.count.astype(jnp.int32),
                mean=state.mean.astype(jnp.float32),
                var=state.var.astype(jnp.float32),
            )

        valid: Array | None = None
        if mask is not None:
            if mask.shape != (batch_size, length):
                raise ValueError(f"mask must have shape {(batch_size, length)}, got {mask.shape}")
            valid = mask.astype(jnp.bool_)

        boundaries: Array | None = None
        if segment_ids is not None:
            if segment_ids.shape != (batch_size, length):
                raise ValueError(
                    f"segment_ids must have shape {(batch_size, length)}, got {segment_ids.shape}"
                )
            # Anchor continuation at the last real token, matching ComplexEMA.
            segment_valid = valid_segment_mask(segment_ids)
            valid = segment_valid if valid is None else valid & segment_valid
            boundaries = segment_boundaries(segment_ids)

        if length == 0:
            return x, initial

        if state is None:
            if self.prior_count > jnp.iinfo(jnp.int32).max - length:
                raise ValueError("TimestepNorm int32 count would overflow")
            initial_count = initial.count
        else:
            if _state_validated:
                initial_count = initial.count
            else:
                max_increment = (
                    jnp.asarray(length, dtype=jnp.int32)
                    if valid is None
                    else valid.astype(jnp.int32).sum(axis=1)
                )
                initial_count = eqx.error_if(
                    initial.count,
                    jnp.any(
                        (initial.count < 0)
                        | (initial.count > jnp.iinfo(jnp.int32).max - max_increment)
                    ),
                    "TimestepNorm count must be non-negative and must not overflow int32",
                )
        initial = NormState(count=initial_count, mean=initial.mean, var=initial.var)

        groups = self.num_groups
        group_size = self.group_size
        x_groups = x.astype(jnp.float32).reshape(batch_size, length, groups, group_size)
        prior_count = prior.count
        prior_mean = prior.mean
        prior_var = prior.var

        moment_groups = (
            x_groups if valid is None else jnp.where(valid[..., None, None], x_groups, 0.0)
        )
        block_mean, block_var = _block_moments(moment_groups)
        if valid is None:
            cumulative_count, cumulative_mean, cumulative_var = _shifted_cumsum_prefix(
                block_mean,
                block_var,
                initial,
            )
        else:
            token_summary = _MomentSummary(
                count=valid[..., None].astype(jnp.int32),
                mean=jnp.where(valid[..., None], block_mean, 0.0),
                m2=jnp.where(valid[..., None], block_var, 0.0),
            )
            if boundaries is None:
                prefix = jax.lax.associative_scan(_merge_m2, token_summary, axis=1)
            else:
                prefix_segmented = jax.lax.associative_scan(
                    _merge_segmented_m2,
                    _SegmentedSummary(
                        count=token_summary.count,
                        mean=token_summary.mean,
                        m2=token_summary.m2,
                        has_reset=boundaries[..., None],
                    ),
                    axis=1,
                )
                prefix = _MomentSummary(
                    prefix_segmented.count,
                    prefix_segmented.mean,
                    prefix_segmented.m2,
                )

            initial_count_expanded = initial.count[:, None, None]
            initial_summary = _MomentSummary(
                count=initial_count_expanded,
                mean=initial.mean[:, None, :],
                m2=initial.var[:, None, :] * initial_count_expanded.astype(jnp.float32),
            )
            cumulative = _merge_m2(initial_summary, prefix)
            cumulative_count = cumulative.count
            cumulative_mean = cumulative.mean
            cumulative_var = jnp.where(
                cumulative_count > 0,
                cumulative.m2 / jnp.maximum(cumulative_count.astype(jnp.float32), 1.0),
                initial.var[:, None, :],
            )

        centered = moment_groups - cumulative_mean[..., None]
        # Preserve released normalization semantics: eps is added at normalization time, with
        # no variance floor or clamp. Incoming cache validation rejects negative state variance.
        normalized = centered * jax.lax.rsqrt(cumulative_var[..., None] + self.eps)
        scale = (self.weight + 1.0).reshape(groups, group_size)
        bias = self.bias.reshape(groups, group_size)
        outputs = normalized * scale + bias
        if valid is not None:
            outputs = jnp.where(valid[..., None, None], outputs, 0.0)

        if segment_ids is None:
            final_count = cumulative_count[:, -1, 0]
            final_mean = cumulative_mean[:, -1]
            final_var = cumulative_var[:, -1]
        else:
            assert valid is not None
            positions = jnp.arange(length, dtype=jnp.int32)[None, :]
            last_valid = jnp.max(jnp.where(valid, positions, -1), axis=1)
            gather_index = jnp.maximum(last_valid, 0)
            final_count = jnp.take_along_axis(
                cumulative_count[..., 0], gather_index[:, None], axis=1
            )[:, 0]
            final_mean = jnp.take_along_axis(
                cumulative_mean,
                gather_index[:, None, None],
                axis=1,
            )[:, 0]
            final_var = jnp.take_along_axis(
                cumulative_var,
                gather_index[:, None, None],
                axis=1,
            )[:, 0]
            has_valid = last_valid >= 0
            final_count = jnp.where(has_valid, final_count, prior_count)
            final_mean = jnp.where(has_valid[:, None], final_mean, prior_mean)
            final_var = jnp.where(has_valid[:, None], final_var, prior_var)

        y = outputs.reshape(batch_size, length, dim).astype(x.dtype)
        return y, NormState(count=final_count, mean=final_mean, var=final_var)
