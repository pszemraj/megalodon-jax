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

from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class RotaryTable(NamedTuple):
    """Derived FP32 rotary factors reusable across model layers."""

    cos: Float[Array, "batch_or_one seq one half_dim"]
    sin: Float[Array, "batch_or_one seq one half_dim"]


class RotaryEmbedding(eqx.Module):
    """Rotary Positional Embedding (RoPE).

    Applies position-dependent rotations to query and key vectors. Standalone
    calls derive cos/sin on demand; the model stack may provide one per-call
    table shared by every layer.

    The rotation treats adjacent dimension pairs as complex numbers, matching
    the released implementation's reshape(..., -1, 2) convention.

    Attributes:
        dim: Per-head dimensionality (must be even).
        base: Exponential base controlling angular step size.
    """

    dim: int = eqx.field(static=True)
    base: float = eqx.field(static=True)

    def __init__(self, dim: int, base: float = 10000.0):
        """Initialize RotaryEmbedding.

        :param int dim: Per-head dimensionality (must be even).
        :param float base: Exponential base for frequency computation.
        :raises ValueError: If dim is not even.
        :return None: None.
        """
        if dim % 2 != 0:
            raise ValueError(f"RotaryEmbedding expects even head dimension, got {dim}")
        self.dim = dim
        self.base = base

    def _inverse_frequencies(self) -> Float[Array, "half_dim"]:
        """Derive fixed frequencies without adding an array leaf to the module."""
        half = self.dim // 2
        exponent = jnp.arange(half, dtype=jnp.float32) / half
        return jnp.power(jnp.asarray(self.base, dtype=jnp.float32), -exponent)

    def table_from_positions(
        self,
        positions: Int[Array, "seq"] | Int[Array, "batch seq"],
    ) -> RotaryTable:
        """Derive FP32 cos/sin factors for explicit token positions.

        A rank-one position vector produces a singleton-batch table that
        broadcasts across rows. Rank-two positions preserve per-row phases.

        :param Int[Array, "seq"] | Int[Array, "batch seq"] positions: Token positions.
        :raises ValueError: If positions is not rank one or two.
        :return RotaryTable: Broadcast-ready FP32 cosine and sine factors.
        """
        if positions.ndim not in (1, 2):
            raise ValueError(f"RoPE positions must have rank 1 or 2, got {positions.shape}")
        inv_freq = self._inverse_frequencies()
        angles = positions.astype(jnp.float32)[..., None] * inv_freq
        if positions.ndim == 1:
            angles = angles[None, ...]
        return RotaryTable(
            cos=jnp.cos(angles)[:, :, None, :],
            sin=jnp.sin(angles)[:, :, None, :],
        )

    def _table(
        self,
        batch_size: int,
        seq_len: int,
        start_index: Int[Array, ""],
        position_ids: Int[Array, "batch seq"] | None,
    ) -> RotaryTable:
        """Resolve implicit or explicit positions into rotary factors.

        :param int batch_size: Number of input sequences.
        :param int seq_len: Number of tokens per sequence.
        :param Int[Array, ""] start_index: Absolute offset for implicit positions.
        :param Int[Array, "batch seq"] | None position_ids: Optional explicit positions.
        :return RotaryTable: Broadcast-ready FP32 cosine and sine factors.
        """
        if position_ids is None:
            positions = jnp.arange(seq_len, dtype=jnp.float32) + start_index.astype(jnp.float32)
        else:
            if position_ids.shape != (batch_size, seq_len):
                raise ValueError(
                    f"position_ids must have shape {(batch_size, seq_len)}, "
                    f"got {position_ids.shape}"
                )
            positions = position_ids
        return self.table_from_positions(positions)

    def __call__(
        self,
        q: Float[Array, "batch seq heads dim"],
        k: Float[Array, "batch seq heads dim"],
        start_index: Int[Array, ""],  # JAX scalar, NOT Python int
        position_ids: Int[Array, "batch seq"] | None = None,
        *,
        table: RotaryTable | None = None,
    ) -> tuple[Float[Array, "batch seq heads dim"], Float[Array, "batch seq heads dim"]]:
        """Apply rotary embedding to query and key tensors.

        :param Float[Array, "batch seq heads dim"] q: Query tensor.
        :param Float[Array, "batch seq heads dim"] k: Key tensor.
        :param Int[Array, ""] start_index: Absolute position offset (JAX scalar).
        :param Int[Array, "batch seq"] | None position_ids: Optional explicit per-token positions.
        :param RotaryTable | None table: Optional factors derived once by the model stack.
        :return tuple[Float[Array, "batch seq heads dim"], Float[Array, "batch seq heads dim"]]: Rotated query and key tensors.
        """
        batch_size = q.shape[0]
        seq_len = q.shape[1]
        if q.shape != k.shape:
            raise ValueError(f"RoPE query/key shapes must match, got {q.shape} and {k.shape}")
        if q.shape[-1] != self.dim:
            raise ValueError(f"RoPE expected final dimension {self.dim}, got {q.shape[-1]}")
        if table is None:
            table = self._table(batch_size, seq_len, start_index, position_ids)
        elif table.cos.shape != table.sin.shape:
            raise ValueError(
                f"RoPE table cosine/sine shapes must match, got "
                f"{table.cos.shape} and {table.sin.shape}"
            )
        expected_suffix = (seq_len, 1, self.dim // 2)
        if table.cos.shape[0] not in (1, batch_size) or table.cos.shape[1:] != expected_suffix:
            raise ValueError(
                "RoPE table must have shape "
                f"(1 or {batch_size}, {seq_len}, 1, {self.dim // 2}), got {table.cos.shape}"
            )
        cos, sin = table

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
