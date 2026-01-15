"""Streaming TimestepNorm for Megalodon JAX.

TimestepNorm is a streaming variant of GroupNorm that computes cumulative
statistics (Welford's algorithm) to avoid leaking future information in
autoregressive models.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, PRNGKeyArray

from megalodon_jax.types import NormState

# Variance floor to prevent division instability in early training
VARIANCE_FLOOR = 1e-6

# Supported input dtypes (fp16 rejected for numerical stability)
_SUPPORTED_DTYPES = (jnp.float32, jnp.bfloat16)


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
            self.weight = jnp.zeros(num_features)
            self.bias = jnp.zeros(num_features)
        else:
            self.weight = None
            self.bias = None

    def __call__(
        self,
        x: Float[Array, "batch seq dim"],
        state: NormState | None = None,
        mask: Bool[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq dim"], NormState]:
        """Normalize x while carrying forward streaming statistics.

        :param Float[Array, "batch seq dim"] x: Input tensor.
        :param NormState | None state: Previous running statistics.
        :param Bool[Array, "batch seq"] | None mask: Valid-token mask.
        :raises ValueError: If input dimension doesn't match num_features.
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
            prev_mean = state.mean
            prev_var = state.var

        # Default mask: all valid
        if mask is None:
            mask = jnp.ones((B, L), dtype=jnp.bool_)

        # Handle empty sequence
        if L == 0:
            new_state = NormState(
                count=prev_count,
                mean=prev_mean.astype(x.dtype),
                var=prev_var.astype(x.dtype),
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
        valid_exp = valid[:, :, None]  # (B, L, 1)

        # Per-group means at each position: (B, L, G)
        group_means = x_groups.mean(axis=-1)

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

        # Output state from final position
        new_count = prev_count + mask.astype(jnp.int32).sum(axis=1)
        mean_out = mean_t[:, -1, :].astype(x.dtype)
        var_out = var_t[:, -1, :].astype(x.dtype)

        new_state = NormState(count=new_count, mean=mean_out, var=var_out)
        return y, new_state
