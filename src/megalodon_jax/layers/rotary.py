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
"""Rotary Positional Embedding for Megalodon JAX."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray


class RotaryEmbedding(eqx.Module):
    """Rotary Positional Embedding (RoPE).

    Applies position-dependent rotations to query and key vectors.
    Computes cos/sin on the fly rather than caching a large table.

    The rotation treats adjacent dimension pairs as complex numbers, matching
    the released implementation's reshape(..., -1, 2) convention.

    Attributes:
        dim: Per-head dimensionality (must be even).
        base: Exponential base controlling angular step size.
    """

    dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)

    def __init__(self, dim: int, base: float = 10000.0, *, key: PRNGKeyArray | None = None):
        """Initialize RotaryEmbedding.

        :param int dim: Per-head dimensionality (must be even).
        :param float base: Exponential base for frequency computation.
        :param PRNGKeyArray | None key: PRNG key (unused).
        :raises ValueError: If dim is not even.
        :return None: None.
        """
        del key  # unused
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding expects even head dimension, got {dim}")
        self.dim = dim
        self.base = base

    def _inverse_frequencies(self) -> Float[Array, "half_dim"]:
        """Derive fixed frequencies without adding an array leaf to the module."""
        half = self.dim // 2
        exponent = jnp.arange(half, dtype=jnp.float32) / half
        return jnp.power(jnp.asarray(self.base, dtype=jnp.float32), -exponent)

    def __call__(
        self,
        q: Float[Array, "batch seq heads dim"],
        k: Float[Array, "batch seq heads dim"],
        start_index: Int[Array, ""],  # JAX scalar, NOT Python int
        position_ids: Int[Array, "batch seq"] | None = None,
    ) -> tuple[Float[Array, "batch seq heads dim"], Float[Array, "batch seq heads dim"]]:
        """Apply rotary embedding to query and key tensors.

        :param Float[Array, "batch seq heads dim"] q: Query tensor.
        :param Float[Array, "batch seq heads dim"] k: Key tensor.
        :param Int[Array, ""] start_index: Absolute position offset (JAX scalar).
        :param Int[Array, "batch seq"] | None position_ids: Optional explicit per-token positions.
        :return tuple[Float[Array, "batch seq heads dim"], Float[Array, "batch seq heads dim"]]: Rotated query and key tensors.
        """
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        if q.shape != k.shape:
            raise ValueError(f"RoPE query/key shapes must match, got {q.shape} and {k.shape}")
        if q.shape[-1] != self.dim:
            raise ValueError(f"RoPE expected final dimension {self.dim}, got {q.shape[-1]}")
        inv_freq = self._inverse_frequencies()

        if position_ids is None:
            # Compute positions: [start_index, start_index+1, ..., start_index+seq_len-1]
            positions = jnp.arange(seq_len, dtype=jnp.float32) + start_index.astype(jnp.float32)
            angles = positions[:, None] * inv_freq[None, :]  # (seq, half_dim)
            cos = jnp.cos(angles)[None, :, None, :]  # (1, seq, 1, half_dim)
            sin = jnp.sin(angles)[None, :, None, :]  # (1, seq, 1, half_dim)
        else:
            positions = position_ids.astype(jnp.float32)
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_ids must have shape {(batch_size, seq_len)}, "
                    f"got {position_ids.shape}"
                )
            angles = positions[:, :, None] * inv_freq[None, None, :]  # (B, seq, half_dim)
            cos = jnp.cos(angles)[:, :, None, :]  # (B, seq, 1, half_dim)
            sin = jnp.sin(angles)[:, :, None, :]  # (B, seq, 1, half_dim)
            cos = jnp.broadcast_to(cos, (batch_size, seq_len, 1, self.dim // 2))
            sin = jnp.broadcast_to(sin, (batch_size, seq_len, 1, self.dim // 2))

        # Match torch.view_as_complex(x.float().reshape(..., -1, 2)).
        half = self.dim // 2
        q_pairs = q.astype(jnp.float32).reshape(*q.shape[:-1], half, 2)
        k_pairs = k.astype(jnp.float32).reshape(*k.shape[:-1], half, 2)
        q1, q2 = q_pairs[..., 0], q_pairs[..., 1]
        k1, k2 = k_pairs[..., 0], k_pairs[..., 1]

        # Apply rotation: (a + ib)(cos + i*sin) = (a*cos - b*sin) + i(b*cos + a*sin)
        q_rot = jnp.stack([q1 * cos - q2 * sin, q2 * cos + q1 * sin], axis=-1)
        k_rot = jnp.stack([k1 * cos - k2 * sin, k2 * cos + k1 * sin], axis=-1)

        return q_rot.reshape(q.shape).astype(q.dtype), k_rot.reshape(k.shape).astype(k.dtype)
