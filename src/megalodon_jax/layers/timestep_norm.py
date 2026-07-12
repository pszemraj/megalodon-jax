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

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray

from megalodon_jax.layers.segments import segment_boundaries, valid_segment_mask
from megalodon_jax.types import NormState


def _block_moments(x: Array) -> tuple[Array, Array]:
    """Return stable population moments over the final feature axis."""
    mean = jnp.mean(x, axis=-1)
    var = jnp.mean(jnp.square(x - mean[..., None]), axis=-1)
    return mean, var


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
        *,
        key: PRNGKeyArray | None = None,
    ):
        """Initialize TimestepNorm.

        :param int num_features: Total feature width.
        :param int | None num_groups: Number of statistic groups. None or
            num_features selects featurewise TimestepNorm.
        :param int prior_count: Number of learned-prior observations.
        :param float eps: Numerical epsilon added to population variance.
        :param bool affine: Compatibility argument; must be true because released
            TimestepNorm is always affine.
        :param PRNGKeyArray | None key: Unused initialization key.
        :raises ValueError: If dimensions, prior count, epsilon, or affine mode are invalid.
        """
        del key
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if prior_count < 0:
            raise ValueError(f"prior_count must be non-negative, got {prior_count}")
        if prior_count > jnp.iinfo(jnp.int32).max:
            raise ValueError("prior_count exceeds the supported int32 state range")
        if eps < 0.0:
            raise ValueError(f"eps must be non-negative, got {eps}")
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
        """Create the source-compatible initial state for a batch."""
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
    ) -> tuple[Float[Array, "batch seq dim"], NormState]:
        """Normalize a sequence and return its final causal statistic state.

        Public masks use True for valid tokens. Segment id zero is padding;
        every contiguous nonzero segment is normalized as an independent sequence.
        Masked outputs are exactly zero after affine transformation.

        :param Float[Array, "batch seq dim"] x: Input activations.
        :param NormState | None state: Optional continuation state.
        :param Bool[Array, "batch seq"] | None mask: Valid-token mask.
        :param Int[Array, "batch seq"] | None segment_ids: Packed-sequence ids.
        :raises ValueError: If shapes are invalid or packed resets are combined with state.
        :raises TypeError: If float16 input is provided.
        :return tuple: Normalized output and final population-statistic state.
        """
        if x.ndim != 3:
            raise ValueError(f"TimestepNorm expects rank-3 input, got shape {x.shape}")
        batch_size, length, dim = x.shape
        if dim != self.num_features:
            raise ValueError(f"TimestepNorm expected {self.num_features} features, got {dim}")
        if x.dtype == jnp.float16:
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

        if mask is None:
            valid = jnp.ones((batch_size, length), dtype=jnp.bool_)
        else:
            if mask.shape != (batch_size, length):
                raise ValueError(f"mask must have shape {(batch_size, length)}, got {mask.shape}")
            valid = mask.astype(jnp.bool_)

        if segment_ids is None:
            boundaries = jnp.zeros((batch_size, length), dtype=jnp.bool_)
        else:
            if segment_ids.shape != (batch_size, length):
                raise ValueError(
                    f"segment_ids must have shape {(batch_size, length)}, got {segment_ids.shape}"
                )
            # Anchor continuation at the last real token, matching ComplexEMA.
            valid = valid & valid_segment_mask(segment_ids)
            boundaries = segment_boundaries(segment_ids)

        if length == 0:
            return x, initial

        if state is None:
            initial_count = initial.count
        else:
            max_increment = valid.astype(jnp.int32).sum(axis=1)
            initial_count = eqx.error_if(
                initial.count,
                jnp.any(initial.count > jnp.iinfo(jnp.int32).max - max_increment),
                "TimestepNorm int32 count would overflow",
            )
        initial = NormState(count=initial_count, mean=initial.mean, var=initial.var)

        groups = self.num_groups
        group_size = self.group_size
        x_groups = x.astype(jnp.float32).reshape(batch_size, length, groups, group_size)
        prior_count = prior.count
        prior_mean = prior.mean
        prior_var = prior.var

        block_mean, block_var = _block_moments(x_groups)
        token_count = valid[..., None].astype(jnp.int32)
        token_mean = jnp.where(valid[..., None], block_mean, 0.0)
        token_var = jnp.where(valid[..., None], block_var, 0.0)

        def merge_welford(
            lhs: tuple[Array, Array, Array],
            rhs: tuple[Array, Array, Array],
        ) -> tuple[Array, Array, Array]:
            """Associatively merge population-variance Welford summaries."""
            lhs_count, lhs_mean, lhs_var = lhs
            rhs_count, rhs_mean, rhs_var = rhs
            count = lhs_count + rhs_count
            lhs_count_f = lhs_count.astype(jnp.float32)
            rhs_count_f = rhs_count.astype(jnp.float32)
            count_f = count.astype(jnp.float32)
            denominator = jnp.maximum(count_f, 1.0)
            delta = rhs_mean - lhs_mean
            mean = lhs_mean + delta * (rhs_count_f / denominator)
            var = (
                lhs_count_f * lhs_var
                + rhs_count_f * rhs_var
                + jnp.square(delta) * lhs_count_f * rhs_count_f / denominator
            ) / denominator
            nonempty = count > 0
            return (
                count,
                jnp.where(nonempty, mean, 0.0),
                jnp.where(nonempty, var, 0.0),
            )

        if segment_ids is None:
            # For the overwhelmingly common unsegmented path, evaluate the
            # same Welford recurrence from vectorized prefix sums. This avoids
            # materializing a log-depth scan tree while retaining the within-
            # token variance term that the earlier implementation omitted.
            initial_count_f = initial.count.astype(jnp.float32)
            valid_f = valid.astype(jnp.float32)
            count_f = initial_count_f[:, None] + jnp.cumsum(valid_f, axis=1)
            count_denominator = jnp.maximum(count_f, 1.0)
            running_sum = initial.mean * initial_count_f[:, None]
            running_sum = running_sum[:, None, :] + jnp.cumsum(
                block_mean * valid_f[..., None], axis=1
            )
            cumulative_mean = jnp.where(
                count_f[..., None] > 0.0,
                running_sum / count_denominator[..., None],
                initial.mean[:, None, :],
            )

            previous_mean = jnp.concatenate(
                (initial.mean[:, None, :], cumulative_mean[:, :-1, :]), axis=1
            )
            delta = block_mean - previous_mean
            delta_after = block_mean - cumulative_mean
            m2_increment = (block_var + delta * delta_after) * valid_f[..., None]
            initial_m2 = initial.var * initial_count_f[:, None]
            cumulative_m2 = initial_m2[:, None, :] + jnp.cumsum(m2_increment, axis=1)
            cumulative_var = jnp.where(
                count_f[..., None] > 0.0,
                cumulative_m2 / count_denominator[..., None],
                initial.var[:, None, :],
            )
            cumulative_count = initial.count[:, None] + jnp.cumsum(valid.astype(jnp.int32), axis=1)
        else:
            elements = (token_count, token_mean, token_var)

            def merge_segmented(
                lhs: tuple[Array, Array, Array, Array],
                rhs: tuple[Array, Array, Array, Array],
            ) -> tuple[Array, Array, Array, Array]:
                """Merge Welford summaries, discarding history at right-hand resets."""
                lhs_count, lhs_mean, lhs_var, lhs_reset = lhs
                rhs_count, rhs_mean, rhs_var, rhs_reset = rhs
                count, mean, var = merge_welford(
                    (lhs_count, lhs_mean, lhs_var),
                    (rhs_count, rhs_mean, rhs_var),
                )
                return (
                    jnp.where(rhs_reset, rhs_count, count),
                    jnp.where(rhs_reset, rhs_mean, mean),
                    jnp.where(rhs_reset, rhs_var, var),
                    lhs_reset | rhs_reset,
                )

            prefix_count, prefix_mean, prefix_var, _ = jax.lax.associative_scan(
                merge_segmented,
                (*elements, boundaries[..., None]),
                axis=1,
            )

            initial_count = initial.count[:, None, None]
            initial_mean = initial.mean[:, None, :]
            initial_var = initial.var[:, None, :]
            cumulative_count, cumulative_mean, cumulative_var = merge_welford(
                (initial_count, initial_mean, initial_var),
                (prefix_count, prefix_mean, prefix_var),
            )
            empty_prefix = prefix_count == 0
            cumulative_count = jnp.where(empty_prefix, initial_count, cumulative_count)
            cumulative_mean = jnp.where(empty_prefix, initial_mean, cumulative_mean)
            cumulative_var = jnp.where(empty_prefix, initial_var, cumulative_var)

        centered = x_groups - cumulative_mean[..., None]
        normalized = centered * jax.lax.rsqrt(cumulative_var[..., None] + self.eps)
        scale = (self.weight + 1.0).reshape(groups, group_size)
        bias = self.bias.reshape(groups, group_size)
        outputs = normalized * scale + bias
        outputs = jnp.where(valid[..., None, None], outputs, 0.0)

        if segment_ids is None:
            final_count = cumulative_count[:, -1]
            final_mean = cumulative_mean[:, -1]
            final_var = cumulative_var[:, -1]
        else:
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
