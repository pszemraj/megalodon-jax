"""Normalization layers for Megalodon JAX."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class RMSNorm(eqx.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input by its RMS value and applies a learnable scale.
    The gamma parameter uses a (gamma + 1) parameterization so that
    initializing gamma to zeros gives an effective scale of 1.

    Attributes:
        dim: Feature dimension.
        eps: Small constant for numerical stability.
        gamma: Learnable scale parameter (shape: dim).
    """

    dim: int = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    gamma: Float[Array, "dim"]

    def __init__(self, dim: int, eps: float = 1e-6, *, key: PRNGKeyArray | None = None):
        """Initialize RMSNorm.

        Args:
            dim: Feature dimension.
            eps: Small constant for numerical stability.
            key: PRNG key (unused, for API consistency).
        """
        del key  # unused
        self.dim = dim
        self.eps = eps
        self.gamma = jnp.zeros(dim)

    def __call__(self, x: Float[Array, "... dim"]) -> Float[Array, "... dim"]:
        """Apply RMS normalization.

        Args:
            x: Input tensor with feature dimension last.

        Returns:
            Normalized tensor with same shape as input.
        """
        # Compute RMS: sqrt(mean(x^2) + eps)
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
        # Normalize and apply scale (gamma + 1 parameterization)
        return x / rms * (self.gamma + 1.0)
