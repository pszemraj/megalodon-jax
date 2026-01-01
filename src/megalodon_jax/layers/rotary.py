"""Rotary Positional Embedding for Megalodon JAX."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


class RotaryEmbedding(eqx.Module):
    """Rotary Positional Embedding (RoPE).

    Applies position-dependent rotations to query and key vectors.
    Computes cos/sin on the fly rather than caching a large table.

    The rotation treats consecutive dimension pairs as complex numbers:
    dimensions [0:half] are real parts, [half:dim] are imaginary parts.

    Attributes:
        dim: Per-head dimensionality (must be even).
        base: Exponential base controlling angular step size.
        inv_freq: Precomputed inverse frequencies.
    """

    dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)
    inv_freq: Float[Array, "half_dim"]

    def __init__(self, dim: int, base: float = 10000.0, *, key: PRNGKeyArray | None = None):
        """Initialize RotaryEmbedding.

        Args:
            dim: Per-head dimensionality (must be even).
            base: Exponential base for frequency computation.
            key: PRNG key (unused, for API consistency).

        Raises:
            ValueError: If dim is not even.
        """
        del key  # unused
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding expects even head dimension, got {dim}")
        self.dim = dim
        self.base = base
        half = dim // 2
        # inv_freq = 1 / (base^(i / half)) for i in [0, half)
        # Equivalent: exp(-i * log(base) / half)
        # Note: JAX's exp() may differ slightly from PyTorch's at float32 precision.
        # This is acceptable as long as the difference is within float32 epsilon bounds.
        self.inv_freq = jnp.exp(jnp.arange(half, dtype=jnp.float32) * (-jnp.log(base) / half))

    def __call__(
        self,
        q: Float[Array, "batch seq heads dim"],
        k: Float[Array, "batch seq heads dim"],
        start_index: Int[Array, ""],  # JAX scalar, NOT Python int
    ) -> tuple[Float[Array, "batch seq heads dim"], Float[Array, "batch seq heads dim"]]:
        """Apply rotary embedding to query and key tensors.

        Args:
            q: Query tensor of shape (batch, seq, heads, dim).
            k: Key tensor of shape (batch, seq, heads, dim).
            start_index: Absolute position offset (must be JAX array scalar).

        Returns:
            Tuple of rotated (q, k) tensors with same shapes.
        """
        seq_len = q.shape[1]

        # Compute positions: [start_index, start_index+1, ..., start_index+seq_len-1]
        positions = jnp.arange(seq_len, dtype=jnp.float32) + start_index

        # Compute angles: (seq, half_dim)
        angles = positions[:, None] * self.inv_freq[None, :]

        # Compute cos/sin in input dtype
        cos = jnp.cos(angles).astype(q.dtype)  # (seq, half_dim)
        sin = jnp.sin(angles).astype(q.dtype)  # (seq, half_dim)

        # Reshape for broadcasting: (1, seq, 1, half_dim)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]

        # Split into real/imag pairs (first half, second half)
        half = self.dim // 2
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]

        # Apply rotation: (a + ib)(cos + i*sin) = (a*cos - b*sin) + i(b*cos + a*sin)
        q_rot = jnp.concatenate([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
        k_rot = jnp.concatenate([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

        return q_rot, k_rot
