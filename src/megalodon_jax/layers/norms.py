"""Normalization layers for Megalodon JAX."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value and optionally applies a learnable scale.
    The gamma parameter uses a (gamma + 1) parameterization so that
    initializing gamma to zeros gives an effective scale of 1.

    Attributes:
        dim: Feature dimension.
        eps: Small constant for numerical stability.
        affine: Whether to include learnable scale parameter.
        gamma: Learnable scale parameter (shape: dim), or None if affine=False.
    """

    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    affine: bool = eqx.field(static=True)
    gamma: Float[Array, "dim"] | None

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        affine: bool = True,
        *,
        key: PRNGKeyArray | None = None,
    ):
        """Initialize RMSNorm.

        :param int dim: Feature dimension.
        :param float eps: Numerical stability epsilon.
        :param bool affine: Whether to include learnable scale.
        :param PRNGKeyArray | None key: PRNG key (unused).
        :return None: None.
        """
        del key  # unused
        self.dim = dim
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = jnp.zeros(dim)
        else:
            self.gamma = None

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Apply RMS normalization.

        :param Float[Array, "... dim"] x: Input tensor with feature dimension last.
        :return Float[Array, "... dim"]: Normalized tensor.
        """
        # Compute RMS in fp32 to avoid bf16 overflow on x**2
        # (bf16 max ~65504, so values > ~256 would overflow when squared)
        x_f32 = x.astype(jnp.float32)
        rms = jnp.sqrt(jnp.mean(x_f32**2, axis=-1, keepdims=True) + self.eps)
        x_normed = (x_f32 / rms).astype(x.dtype)
        # Apply scale if affine (cast gamma to input dtype to preserve bf16)
        if self.affine and self.gamma is not None:
            scale = (self.gamma + 1.0).astype(x.dtype)
            return x_normed * scale
        return x_normed
