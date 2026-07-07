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
"""Streaming TimestepNorm for Megalodon JAX.

TimestepNorm is a streaming variant of GroupNorm that computes cumulative
statistics (Welford's algorithm) to avoid leaking future information in
autoregressive models.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.types import NormState

# Variance floor to prevent division instability in early training
VARIANCE_FLOOR = 1e-6


class TimestepNorm(eqx.Module):
    """Streaming group-wise normalization across time with optional state.

    At position t, normalizes using only statistics from positions 0..t,
    enabling causal/autoregressive processing.

    Attributes:
        num_features: Total number of feature channels D.
        num_groups: Number of groups for group-wise statistics.
        group_size: Features per group (num_features // num_groups).
        eps: Numerical epsilon for variance stability.
        affine: Whether to include learnable affine parameters.
        weight: Scale parameter (effective scale = weight + 1.0), or None if affine=False.
        bias: Shift parameter, or None if affine=False.
    """

    num_features: int = eqx.field(static=True)
    num_groups: int = eqx.field(static=True)
    group_size: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)
    weight: Float[Array, "dim"] | None
    bias: Float[Array, "dim"] | None

    def __init__(
        self,
        num_features: int,
        num_groups: int,
        eps: float = 1e-5,
        affine: bool = True,
        *,
        key: PRNGKeyArray | None = None,
    ):
        """Initialize TimestepNorm.

        :param int num_features: Total number of feature channels.
        :param int num_groups: Number of groups for statistics computation.
        :param float eps: Numerical epsilon for variance stability.
        :param bool affine: Whether to include learnable affine parameters.
        :param PRNGKeyArray | None key: PRNG key (unused).
        :raises ValueError: If num_features is not divisible by num_groups.
        :return None: None.
        """
        del key  # unused
        if num_features % num_groups != 0:
            raise ValueError(
                f"num_features ({num_features}) must be divisible by num_groups ({num_groups})"
            )
        self.num_features = num_features
        self.num_groups = num_groups
        self.group_size = num_features // num_groups
        self.eps = eps
        self.affine = affine
        if affine:
            # Initialize weight to zeros (effective scale = weight + 1.0 = 1.0)
            self.weight = jnp.zeros(num_features, dtype=jnp.float32)
            self.bias = jnp.zeros(num_features, dtype=jnp.float32)
        else:
            self.weight = None
            self.bias = None

    def __call__(
        self,
        x: Float[Array, "batch seq dim"],
        state: NormState | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
        segment_ids: Int[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq dim"], NormState]:
        """Normalize x while carrying forward streaming statistics.

        When segment_ids is given (packed sequences), the running statistics
        reset at each segment boundary so every packed document normalizes
        exactly as if run alone with a fresh state. The returned state is
        anchored at each row's last valid (non-padding, unmasked) token, so
        trailing padding does not blank it; rows with no valid tokens return
        the fresh-state baseline (count=0, mean=0, var=1).

        :param Float[Array, "batch seq dim"] x: Input tensor.
        :param NormState | None state: Previous running statistics.
        :param Bool[Array, "batch seq"] | None mask: Valid-token mask.
        :param Int[Array, "batch seq"] | None segment_ids: Optional per-token segment IDs
            (0 = padding) for packed-sequence stat resets. Training-only:
            incompatible with an incoming state.
        :raises ValueError: If input dimension doesn't match num_features, or if
            segment_ids is combined with an incoming state.
        :raises TypeError: If input dtype is float16.
        :return tuple[Float[Array, "batch seq dim"], NormState]: Normalized output and state.
        """
        B, L, D = x.shape
        G = self.num_groups
        gs = self.group_size

        # Reject fp16 for numerical stability (matches PyTorch reference)
        if x.dtype == jnp.float16:
            raise TypeError(
                "TimestepNorm does not support float16 inputs due to numerical "
                "stability concerns. Use float32 or bfloat16 instead."
            )

        if segment_ids is not None and state is not None:
            raise ValueError(
                "segment_ids is not supported together with an incoming NormState "
                "(state). Packed-sequence resets are a training-only, non-streaming "
                "feature."
            )

        if D != self.num_features:
            raise ValueError(
                f"TimestepNorm expected input with {self.num_features} features, "
                f"got {D}. Input shape: {tuple(x.shape)}"
            )

        # Initialize state if not provided
        if state is None:
            prev_count = jnp.zeros(B, dtype=jnp.int32)
            prev_mean = jnp.zeros((B, G), dtype=jnp.float32)
            prev_var = jnp.ones((B, G), dtype=jnp.float32)
        else:
            prev_count = state.count
            prev_mean = state.mean.astype(jnp.float32)
            prev_var = state.var.astype(jnp.float32)

        # Default mask: all valid
        if mask is None:
            mask = jnp.ones((B, L), dtype=jnp.bool_)

        # Handle empty sequence
        if L == 0:
            new_state = NormState(
                count=prev_count,
                mean=prev_mean,
                var=prev_var,
            )
            return x, new_state

        # Always compute statistics in float32 for numerical stability
        stats_dtype = jnp.float32

        # Reshape to (B, L, G, gs) for per-group statistics
        x_groups = x.reshape(B, L, G, gs).astype(stats_dtype)
        prev_mean_f = prev_mean.astype(stats_dtype)
        prev_var_f = prev_var.astype(stats_dtype)
        prev_count_f = prev_count.astype(stats_dtype)

        # Valid mask for weighting
        valid = mask.astype(stats_dtype)  # (B, L)
        if segment_ids is not None:
            # Padding (segment 0) must not contribute to statistics
            valid = valid * (segment_ids > 0).astype(stats_dtype)
        valid_exp = valid[:, :, None]  # (B, L, 1)

        # Per-group means at each position: (B, L, G)
        group_means = x_groups.mean(axis=-1)

        if segment_ids is None:
            # Cumulative sums for vectorized Welford
            prev_sum = prev_mean_f * prev_count_f[:, None]  # (B, G)
            cumsum_means = jnp.cumsum(group_means * valid_exp, axis=1)  # (B, L, G)
            sum_t = prev_sum[:, None, :] + cumsum_means  # (B, L, G)

            # Cumulative valid counts
            count_t = prev_count_f[:, None] + jnp.cumsum(valid, axis=1)  # (B, L)
            count_clamped = jnp.maximum(count_t, 1.0)

            # Mean at each position
            mean_t = jnp.where(
                count_t[:, :, None] > 0.0,
                sum_t / count_clamped[:, :, None],
                prev_mean_f[:, None, :],
            )  # (B, L, G)

            # Welford variance: use delta from previous mean and current mean
            # mean_prev[t] = mean_t[t-1] for t>0, prev_mean for t=0
            mean_prev = jnp.concatenate(
                [prev_mean_f[:, None, :], mean_t[:, :-1, :]], axis=1
            )  # (B, L, G)
            delta = group_means - mean_prev
            delta2 = group_means - mean_t

            # M2 accumulator
            prev_count_clamped = jnp.maximum(prev_count_f, 1.0)
            prev_m2 = prev_var_f * prev_count_clamped[:, None]  # (B, G)
            delta_term = delta * delta2 * valid_exp  # (B, L, G)
            m2_t = prev_m2[:, None, :] + jnp.cumsum(delta_term, axis=1)  # (B, L, G)

            # Variance at each position
            var_t = jnp.where(
                count_t[:, :, None] > 0.0,
                m2_t / count_clamped[:, :, None],
                prev_var_f[:, None, :],
            )  # (B, L, G)

            # Floor variance to prevent instability
            var_t = jnp.maximum(var_t, VARIANCE_FLOOR)
        else:
            # Segment-local Welford: subtract each segment's pre-start cumulative
            # value so statistics restart at every boundary, reproducing exactly
            # a fresh state (count=0, mean=0, var=1 <=> M2 baseline 1.0) per doc.
            is_boundary = jnp.concatenate(
                [
                    jnp.ones((B, 1), dtype=jnp.bool_),
                    segment_ids[:, 1:] != segment_ids[:, :-1],
                ],
                axis=1,
            )  # (B, L)

            def _segmented_cumsum(z: Float[Array, "batch seq *rest"]) -> Float[Array, "..."]:
                """Inclusive cumsum that restarts at segment boundaries.

                Uses a reset-carrying associative scan rather than subtracting
                a global cumsum at each boundary: the subtraction is exact
                algebraically but catastrophically cancels in fp32 once earlier
                segments' magnitudes dwarf the local sums (and it leaves a
                gradient path across the boundary). The reset flag hard-blocks
                other segments' values from ever entering a segment's sum.

                :param Float[Array, "batch seq *rest"] z: Per-position addends.
                :return Float[Array, "..."]: Segment-local inclusive cumsum.
                """
                flags = is_boundary if z.ndim == 2 else is_boundary[:, :, None]
                flags = jnp.broadcast_to(flags, z.shape)

                def combine(
                    left: tuple[Float[Array, "..."], Bool[Array, "..."]],
                    right: tuple[Float[Array, "..."], Bool[Array, "..."]],
                ) -> tuple[Float[Array, "..."], Bool[Array, "..."]]:
                    """Compose two segment-sum elements (earlier left, later right).

                    :param tuple left: (partial sum, contains-reset flag) of earlier span.
                    :param tuple right: (partial sum, contains-reset flag) of later span.
                    :return tuple: Combined (sum, flag) for the joined span.
                    """
                    v_l, f_l = left
                    v_r, f_r = right
                    return jnp.where(f_r, v_r, v_l + v_r), f_l | f_r

                vals, _ = jax.lax.associative_scan(combine, (z, flags), axis=1)
                return vals

            count_t = _segmented_cumsum(valid)  # (B, L)
            count_clamped = jnp.maximum(count_t, 1.0)

            # Segment-local mean (zero while count is 0, matching fresh state)
            sum_t = _segmented_cumsum(group_means * valid_exp)
            mean_t = sum_t / count_clamped[:, :, None]  # (B, L, G)

            # mean_prev[t] = mean_t[t-1] within a segment, 0 at segment starts
            mean_t_shifted = jnp.concatenate(
                [jnp.zeros((B, 1, G), dtype=stats_dtype), mean_t[:, :-1, :]], axis=1
            )
            mean_prev = jnp.where(is_boundary[:, :, None], 0.0, mean_t_shifted)
            delta = group_means - mean_prev
            delta2 = group_means - mean_t

            # M2 baseline is 1.0 per segment: fresh state has var=1, count=0,
            # so prev_m2 = prev_var * max(prev_count, 1) = 1
            delta_term = delta * delta2 * valid_exp  # (B, L, G)
            m2_t = 1.0 + _segmented_cumsum(delta_term)  # (B, L, G)

            # Variance (count==0 yields m2/1 = 1.0, matching fresh state)
            var_t = jnp.maximum(m2_t / count_clamped[:, :, None], VARIANCE_FLOOR)

        # Normalize: (B, L, G, gs)
        mean_b = mean_t[:, :, :, None]  # (B, L, G, 1)
        var_b = var_t[:, :, :, None]  # (B, L, G, 1)
        x_hat = (x_groups - mean_b) * jax.lax.rsqrt(var_b + self.eps)

        # Apply affine transform with +1 reparameterization (if affine=True)
        if self.affine and self.weight is not None and self.bias is not None:
            scale = (self.weight + 1.0).reshape(1, 1, G, gs).astype(stats_dtype)
            bias = self.bias.reshape(1, 1, G, gs).astype(stats_dtype)
            y = (x_hat * scale + bias).reshape(B, L, D).astype(x.dtype)
        else:
            y = x_hat.reshape(B, L, D).astype(x.dtype)

        # Output state from final position. Under segmentation the trajectory is
        # local to the last segment, so the count must be too - a whole-row count
        # would corrupt the Welford sufficient statistic (m2 = var * count).
        if segment_ids is None:
            new_count = prev_count + mask.astype(jnp.int32).sum(axis=1)
            mean_out = mean_t[:, -1, :]
            var_out = var_t[:, -1, :]
        else:
            # Anchor at each row's last real token: trailing padding (id 0)
            # starts its own run, so stats at position L-1 would be the
            # fresh-reset baseline instead of the last document's.
            positions = jnp.arange(L, dtype=jnp.int32)
            last_valid = jnp.max(jnp.where(valid > 0, positions[None, :], -1), axis=1)  # (B,)
            has_valid = last_valid >= 0
            idx = jnp.maximum(last_valid, 0)
            batch_idx = jnp.arange(B)
            new_count = jnp.where(has_valid, count_t[batch_idx, idx], 0.0).astype(jnp.int32)
            mean_out = jnp.where(has_valid[:, None], mean_t[batch_idx, idx], 0.0)
            var_out = jnp.where(has_valid[:, None], var_t[batch_idx, idx], 1.0)

        new_state = NormState(count=new_count, mean=mean_out, var=var_out)
        return y, new_state
