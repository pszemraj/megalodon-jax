"""NumPy/float64 reference for the paper-equation TimestepNorm."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PaperNormState:
    """Token-block count and population moments for one batch."""

    count: np.ndarray
    mean: np.ndarray
    var: np.ndarray


def _merge_block(
    count: int,
    mean: np.ndarray,
    var: np.ndarray,
    block: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge one equal-sized group block using float64 Welford algebra."""
    block_mean = np.mean(block, axis=-1, dtype=np.float64)
    block_var = np.mean(np.square(block - block_mean[..., None]), axis=-1, dtype=np.float64)
    next_count = count + 1
    delta = block_mean - mean
    next_mean = mean + delta / float(next_count)
    next_var = (
        float(count) * var + block_var + np.square(delta) * float(count) / float(next_count)
    ) / float(next_count)
    return next_count, next_mean, next_var


def timestep_norm_paper(
    values: np.ndarray,
    *,
    groups: int,
    eps: float,
    weight: np.ndarray,
    bias: np.ndarray,
    state: PaperNormState | None = None,
    prior_count: int = 0,
    prior_mean: np.ndarray | None = None,
    prior_var: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, PaperNormState, np.ndarray]:
    """Evaluate cumulative scalar moments and plus-one affine storage."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected rank-3 input, got {x.shape}")
    batch, length, features = x.shape
    if features % groups:
        raise ValueError(f"features ({features}) must be divisible by groups ({groups})")
    group_size = features // groups
    grouped = x.reshape(batch, length, groups, group_size)
    scale = np.asarray(weight, dtype=np.float64).reshape(groups, group_size) + 1.0
    offset = np.asarray(bias, dtype=np.float64).reshape(groups, group_size)

    base_count = np.full((batch,), prior_count, dtype=np.int64)
    base_mean = np.zeros((batch, groups), dtype=np.float64)
    base_var = np.ones((batch, groups), dtype=np.float64)
    if prior_mean is not None:
        if prior_var is None:
            raise ValueError("prior_var is required with prior_mean")
        base_mean = np.broadcast_to(
            np.asarray(prior_mean, dtype=np.float64), (batch, groups)
        ).copy()
        base_var = np.broadcast_to(np.asarray(prior_var, dtype=np.float64), (batch, groups)).copy()

    if state is None:
        count = base_count.copy()
        mean = base_mean.copy()
        var = base_var.copy()
    else:
        count = np.asarray(state.count, dtype=np.int64).copy()
        mean = np.asarray(state.mean, dtype=np.float64).copy()
        var = np.asarray(state.var, dtype=np.float64).copy()

    valid = np.ones((batch, length), dtype=np.bool_)
    if mask is not None:
        valid &= np.asarray(mask, dtype=np.bool_)
    segments = None if segment_ids is None else np.asarray(segment_ids, dtype=np.int64)
    if segments is not None:
        if state is not None:
            raise ValueError("packed reference does not accept continuation state")
        valid &= segments > 0

    output = np.zeros_like(grouped, dtype=np.float64)
    last_count = base_count.copy() if segments is not None else count.copy()
    last_mean = base_mean.copy() if segments is not None else mean.copy()
    last_var = base_var.copy() if segments is not None else var.copy()
    for row in range(batch):
        for timestep in range(length):
            if segments is not None and (
                timestep == 0 or segments[row, timestep] != segments[row, timestep - 1]
            ):
                count[row] = base_count[row]
                mean[row] = base_mean[row]
                var[row] = base_var[row]
            if not valid[row, timestep]:
                continue
            count[row], mean[row], var[row] = _merge_block(
                int(count[row]), mean[row], var[row], grouped[row, timestep]
            )
            normalized = (grouped[row, timestep] - mean[row, :, None]) / np.sqrt(
                var[row, :, None] + eps
            )
            output[row, timestep] = normalized * scale + offset
            last_count[row] = count[row]
            last_mean[row] = mean[row]
            last_var[row] = var[row]

    final = (
        PaperNormState(last_count, last_mean, last_var)
        if segments is not None
        else PaperNormState(count, mean, var)
    )
    return output.reshape(batch, length, features), final, valid


def central_difference(
    function: Callable[[np.ndarray], float],
    value: np.ndarray,
    step: float,
) -> np.ndarray:
    """Evaluate a central finite-difference gradient for a small array."""
    base = np.asarray(value, dtype=np.float64)
    gradient = np.empty_like(base)
    for index in np.ndindex(base.shape):
        positive = base.copy()
        negative = base.copy()
        positive[index] += step
        negative[index] -= step
        gradient[index] = (function(positive) - function(negative)) / (2.0 * step)
    return gradient
