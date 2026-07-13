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
"""Shared segment-metadata helpers for packed-sequence training.

These define the single boundary semantics used everywhere segment_ids are
consumed: attention masking/chunk re-anchoring, ComplexEMA state resets, and
TimestepNorm statistic resets all derive from the same predicate, so packed
documents stay in lockstep across the three mechanisms.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int


def valid_segment_mask(segment_ids: Int[Array, "batch seq"]) -> Bool[Array, "batch seq"]:
    """Return the shared validity predicate for positive packed-sequence IDs.

    :param Int[Array, "batch seq"] segment_ids: Per-token segment IDs, with zero as padding.
    :return Bool[Array, "batch seq"]: True for tokens belonging to a real segment.
    """
    return segment_ids > 0


def segment_boundaries(segment_ids: Int[Array, "batch seq"]) -> Bool[Array, "batch seq"]:
    """Mark positions where a new contiguous segment run starts.

    Position 0 always starts a run; later positions start one when their
    segment id differs from the previous position's. Padding (id 0) forms
    runs like any other id; callers exclude it via validity masks.

    :param Int[Array, "batch seq"] segment_ids: Per-token segment IDs (0 = padding).
    :return Bool[Array, "batch seq"]: True where a new run starts.
    """
    B, L = segment_ids.shape
    if L == 0:
        return jnp.zeros((B, 0), dtype=jnp.bool_)
    return jnp.concatenate(
        [
            jnp.ones((B, 1), dtype=jnp.bool_),
            segment_ids[:, 1:] != segment_ids[:, :-1],
        ],
        axis=1,
    )


def segment_runs_and_local_positions(
    segment_ids: Int[Array, "batch seq"],
) -> tuple[Int[Array, "batch seq"], Int[Array, "batch seq"]]:
    """Compute contiguous-run indices and run-local positions.

    Run indices: a packer may legally reuse a positive id for non-adjacent
    documents (e.g. ``[1, 1, 2, 2, 1, 1]``); comparing raw ids for equality
    would let the later run attend back to the earlier one. Run indices only
    compare equal within a single contiguous run, matching the boundary-based
    reset semantics of ComplexEMA and TimestepNorm.

    Local positions: offsets from each run's first token. Dividing by the
    chunk size re-anchors attention chunk boundaries at each run start, and
    they double as the default RoPE positions for packed rows, so a document
    that begins mid-chunk gets the same block-diagonal pattern and rotary
    phases it would have running alone.

    :param jax.Array segment_ids: Raw per-token segment ids of shape (batch, seq).
    :return tuple: (run indices starting at 1, position offset within the run).
    """
    boundaries = segment_boundaries(segment_ids)
    run_ids = jnp.cumsum(boundaries.astype(segment_ids.dtype), axis=1)
    positions = jnp.arange(segment_ids.shape[1], dtype=segment_ids.dtype)[None, :]
    run_starts = jax.lax.cummax(jnp.where(boundaries, positions, 0), axis=1)
    return run_ids, positions - run_starts
