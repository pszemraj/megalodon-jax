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

        carry = (
            initial.count,
            initial.mean,
            initial.var,
            initial.count,
            initial.mean,
            initial.var,
        )

        def step(
            current: tuple[Array, Array, Array, Array, Array, Array],
            inputs: tuple[Array, Array, Array],
        ) -> tuple[tuple[Array, Array, Array, Array, Array, Array], Array]:
            count, mean, var, last_count, last_mean, last_var = current
            x_t, valid_t, boundary_t = inputs

            count = jnp.where(boundary_t, prior_count, count)
            mean = jnp.where(boundary_t[:, None], prior_mean, mean)
            var = jnp.where(boundary_t[:, None], prior_var, var)

            block_mean = jnp.mean(x_t, axis=-1)
            block_var = jnp.mean(jnp.square(x_t - block_mean[..., None]), axis=-1)
            count_f = count.astype(jnp.float32)
            next_count_f = count_f + 1.0
            delta = block_mean - mean
            merged_mean = mean + delta / next_count_f[:, None]
            merged_var = (
                count_f[:, None] * var
                + block_var
                + jnp.square(delta) * count_f[:, None] / next_count_f[:, None]
            ) / next_count_f[:, None]

            next_count = jnp.where(valid_t, count + 1, count)
            next_mean = jnp.where(valid_t[:, None], merged_mean, mean)
            next_var = jnp.where(valid_t[:, None], merged_var, var)

            centered = x_t - next_mean[..., None]
            normalized = centered * jax.lax.rsqrt(next_var[..., None] + self.eps)
            scale = (self.weight + 1.0).reshape(groups, group_size)
            bias = self.bias.reshape(groups, group_size)
            output = normalized * scale + bias
            output = jnp.where(valid_t[:, None, None], output, 0.0)

            last_count = jnp.where(valid_t, next_count, last_count)
            last_mean = jnp.where(valid_t[:, None], next_mean, last_mean)
            last_var = jnp.where(valid_t[:, None], next_var, last_var)
            next_carry = (
                next_count,
                next_mean,
                next_var,
                last_count,
                last_mean,
                last_var,
            )
            return next_carry, output

        final_carry, outputs = jax.lax.scan(
            step,
            carry,
            (
                jnp.swapaxes(x_groups, 0, 1),
                jnp.swapaxes(valid, 0, 1),
                jnp.swapaxes(boundaries, 0, 1),
            ),
        )
        _, _, _, final_count, final_mean, final_var = final_carry
        y = jnp.swapaxes(outputs, 0, 1).reshape(batch_size, length, dim).astype(x.dtype)
        return y, NormState(count=final_count, mean=final_mean, var=final_var)
