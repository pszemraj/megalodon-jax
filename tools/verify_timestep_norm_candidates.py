#!/usr/bin/env python3
"""Validate parallel TimestepNorm candidates against a float64 paper oracle.

This standalone harness deliberately uses NumPy, rather than the production
TimestepNorm implementation, as its numerical oracle. It exercises adversarial
activation distributions, priors, continuation state, masks, packed resets,
BF16 inputs, and differentiable behavior. Candidate implementations are loaded
from ``tools.timestep_norm_candidates.CANDIDATES``.

Example:

    conda run --name mega-jax python tools/verify_timestep_norm_candidates.py \
      --output local-scratch/timestep-norm-candidate-verification.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _bootstrap_repo(argv: list[str]) -> Path:
    """Put the requested checkout and its source tree on ``sys.path``."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    known, _ = parser.parse_known_args(argv)
    root = known.repo_root.resolve()
    if not (root / "src" / "megalodon_jax").is_dir():
        raise SystemExit(f"not a megalodon-jax checkout: {root}")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    return root


REPO_ROOT = _bootstrap_repo(sys.argv[1:])

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from megalodon_jax.layers.timestep_norm import TimestepNorm  # noqa: E402
from megalodon_jax.types import NormState  # noqa: E402
from tools import timestep_norm_candidates  # noqa: E402

Array = jax.Array
Candidate = Callable[
    [TimestepNorm, Array, NormState | None, Array | None, Array | None],
    tuple[Array, NormState],
]


def _production_timestep_norm(
    norm: TimestepNorm,
    x: Array,
    state: NormState | None,
    mask: Array | None,
    segment_ids: Array | None,
) -> tuple[Array, NormState]:
    """Evaluate the selected production implementation through its public API."""
    return norm(x, state=state, mask=mask, segment_ids=segment_ids)


@dataclass(frozen=True)
class OracleState:
    """NumPy representation of the released token-block statistic state."""

    count: np.ndarray
    mean: np.ndarray
    var: np.ndarray


@dataclass(frozen=True)
class CaseSpec:
    """One deterministic adversarial forward/state case."""

    name: str
    family: str
    values: np.ndarray
    dtype: str = "float32"
    prior_count: int = 0
    prior_mean: np.ndarray | None = None
    prior_logv: np.ndarray | None = None
    state: OracleState | None = None
    mask: np.ndarray | None = None
    segment_ids: np.ndarray | None = None
    output_atol: float = 2e-3
    output_rtol: float = 2e-3


@dataclass(frozen=True)
class ErrorStats:
    """Maximum absolute and elementwise relative error."""

    max_abs: float
    max_relative: float


def _git_revision(repo: Path) -> str:
    """Return the checkout revision without making archives unverifiable."""
    fallback = os.environ.get("MEGALODON_JAX_SOURCE_REVISION", "unknown")
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return fallback
    lines = result.stdout.splitlines()
    if len(lines) != 2 or Path(lines[0]).resolve() != repo.resolve():
        return fallback
    return lines[1].strip()


def _jsonable(value: Any) -> Any:
    """Convert NumPy/JAX values and dataclasses to JSON-compatible objects."""
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    return value


def _error_stats(actual: np.ndarray, expected: np.ndarray) -> ErrorStats:
    """Return finite maximum absolute and relative errors."""
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    difference = np.abs(actual64 - expected64)
    if difference.size == 0:
        return ErrorStats(0.0, 0.0)
    finite = np.isfinite(difference)
    if not np.all(finite):
        return ErrorStats(float("inf"), float("inf"))
    denominator = np.maximum(np.abs(expected64), 1e-12)
    return ErrorStats(float(np.max(difference)), float(np.max(difference / denominator)))


def _merge_block(
    count: int,
    mean: np.ndarray,
    var: np.ndarray,
    block: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Merge one equal-sized feature block with float64 Welford algebra."""
    block_mean = np.mean(block, axis=-1, dtype=np.float64)
    block_var = np.mean(np.square(block - block_mean[..., None]), axis=-1, dtype=np.float64)
    next_count = count + 1
    delta = block_mean - mean
    merged_mean = mean + delta / float(next_count)
    merged_var = (
        float(count) * var + block_var + np.square(delta) * float(count) / float(next_count)
    ) / float(next_count)
    return next_count, merged_mean, merged_var


def numpy_timestep_norm(
    values: np.ndarray,
    *,
    groups: int,
    eps: float,
    weight: np.ndarray,
    bias: np.ndarray,
    prior_count: int,
    prior_mean: np.ndarray | None,
    prior_logv: np.ndarray | None,
    state: OracleState | None = None,
    mask: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, OracleState, np.ndarray]:
    """Evaluate the paper-equation causal normalization in NumPy float64.

    The stored count is the number of equal-sized token blocks. Each block's
    population variance includes within-token feature variance, making this
    algebraically identical to treating every scalar observed so far equally.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"oracle expects rank-3 input, got {x.shape}")
    batch, length, features = x.shape
    if features % groups:
        raise ValueError(f"features ({features}) must be divisible by groups ({groups})")
    group_size = features // groups
    grouped = x.reshape(batch, length, groups, group_size)
    weight64 = np.asarray(weight, dtype=np.float64).reshape(groups, group_size)
    bias64 = np.asarray(bias, dtype=np.float64).reshape(groups, group_size)

    if prior_mean is None:
        base_mean = np.zeros((batch, groups), dtype=np.float64)
        base_var = np.ones((batch, groups), dtype=np.float64)
    else:
        if prior_logv is None:
            raise ValueError("prior_logv is required with prior_mean")
        base_mean = np.broadcast_to(
            np.asarray(prior_mean, dtype=np.float64), (batch, groups)
        ).copy()
        base_var = np.broadcast_to(
            np.exp(np.asarray(prior_logv, dtype=np.float64)),
            (batch, groups),
        ).copy()
    base_count = np.full((batch,), prior_count, dtype=np.int64)

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
            raise ValueError("packed oracle does not accept continuation state")
        valid &= segments > 0

    output = np.zeros_like(grouped, dtype=np.float64)
    last_count = base_count.copy() if segments is not None else count.copy()
    last_mean = base_mean.copy() if segments is not None else mean.copy()
    last_var = base_var.copy() if segments is not None else var.copy()
    for batch_index in range(batch):
        for time_index in range(length):
            if segments is not None and (
                time_index == 0
                or segments[batch_index, time_index] != segments[batch_index, time_index - 1]
            ):
                count[batch_index] = base_count[batch_index]
                mean[batch_index] = base_mean[batch_index]
                var[batch_index] = base_var[batch_index]
            if not valid[batch_index, time_index]:
                continue
            next_count, next_mean, next_var = _merge_block(
                int(count[batch_index]),
                mean[batch_index],
                var[batch_index],
                grouped[batch_index, time_index],
            )
            count[batch_index] = next_count
            mean[batch_index] = next_mean
            var[batch_index] = next_var
            normalized = (grouped[batch_index, time_index] - next_mean[:, None]) / np.sqrt(
                next_var[:, None] + eps
            )
            output[batch_index, time_index] = normalized * (weight64 + 1.0) + bias64
            last_count[batch_index] = next_count
            last_mean[batch_index] = next_mean
            last_var[batch_index] = next_var

    final = (
        OracleState(last_count, last_mean, last_var)
        if segments is not None
        else OracleState(count, mean, var)
    )
    return output.reshape(batch, length, features), final, valid


def _quantize_input(values: np.ndarray, dtype: str) -> tuple[Array, np.ndarray]:
    """Return a JAX input and the exact represented values for the oracle."""
    jax_dtype = {"float32": jnp.float32, "bfloat16": jnp.bfloat16}[dtype]
    array = jnp.asarray(values, dtype=jax_dtype)
    represented = np.asarray(array.astype(jnp.float32), dtype=np.float64)
    return array, represented


def _make_norm(case: CaseSpec, features: int, groups: int) -> TimestepNorm:
    """Construct a nontrivially affine norm for a case."""
    norm = TimestepNorm(features, groups, prior_count=case.prior_count, eps=1e-5)
    weight = jnp.linspace(-0.15, 0.2, features, dtype=jnp.float32)
    bias = jnp.linspace(0.07, -0.05, features, dtype=jnp.float32)
    norm = eqx.tree_at(lambda item: item.weight, norm, weight)
    norm = eqx.tree_at(lambda item: item.bias, norm, bias)
    if case.prior_mean is not None:
        if norm.prior_mean is None or norm.prior_logv is None or case.prior_logv is None:
            raise ValueError(f"case {case.name} has incompatible prior parameters")
        norm = eqx.tree_at(
            lambda item: item.prior_mean,
            norm,
            jnp.asarray(case.prior_mean, dtype=jnp.float32),
        )
        norm = eqx.tree_at(
            lambda item: item.prior_logv,
            norm,
            jnp.asarray(case.prior_logv, dtype=jnp.float32),
        )
    return norm


def _jax_state(state: OracleState | None) -> NormState | None:
    """Convert an oracle continuation state to the candidate representation."""
    if state is None:
        return None
    return NormState(
        count=jnp.asarray(state.count, dtype=jnp.int32),
        mean=jnp.asarray(state.mean, dtype=jnp.float32),
        var=jnp.asarray(state.var, dtype=jnp.float32),
    )


def _case_tolerances(case: CaseSpec, expected: OracleState) -> dict[str, float]:
    """Choose strict tolerances that account for output/storage quantization."""
    mean_scale = max(float(np.max(np.abs(expected.mean))), 1.0)
    var_scale = max(float(np.max(np.abs(expected.var))), 1.0)
    # Around 1e6, even a stable FP32 Welford update quantizes token-block means
    # at 0.0625 increments. Retain a useful one-percent variance bound while
    # still catching raw-moment cancellation and negative variance.
    extreme_offset = float(np.max(np.abs(case.values))) >= 1e5
    return {
        "output_atol": case.output_atol,
        "output_rtol": case.output_rtol,
        "mean_atol": max(5e-5, 8.0 * float(np.spacing(np.float32(mean_scale)))),
        "mean_rtol": 2e-6,
        "var_atol": max(5e-4, 2e-4 * var_scale),
        "var_rtol": 1e-2 if extreme_offset else 2e-4,
    }


def _evaluate_case(
    candidate_name: str,
    candidate: Candidate,
    case: CaseSpec,
    *,
    groups: int,
) -> dict[str, Any]:
    """Compare one candidate invocation with the independent oracle."""
    features = case.values.shape[-1]
    norm = _make_norm(case, features, groups)
    x, represented = _quantize_input(case.values, case.dtype)
    state = _jax_state(case.state)
    mask = None if case.mask is None else jnp.asarray(case.mask, dtype=jnp.bool_)
    segments = None if case.segment_ids is None else jnp.asarray(case.segment_ids, dtype=jnp.int32)
    actual_y, actual_state = candidate(norm, x, state, mask, segments)
    prefix_variance = timestep_norm_candidates.candidate_prefix_variance(
        candidate_name,
        norm,
        x,
        state,
        mask,
        segments,
    )
    actual_y, actual_state, prefix_variance = jax.block_until_ready(
        (actual_y, actual_state, prefix_variance)
    )

    expected_y, expected_state, valid = numpy_timestep_norm(
        represented,
        groups=groups,
        eps=norm.eps,
        weight=np.asarray(norm.weight),
        bias=np.asarray(norm.bias),
        prior_count=norm.prior_count,
        prior_mean=None if norm.prior_mean is None else np.asarray(norm.prior_mean),
        prior_logv=None if norm.prior_logv is None else np.asarray(norm.prior_logv),
        state=case.state,
        mask=case.mask,
        segment_ids=case.segment_ids,
    )
    actual_output = np.asarray(actual_y.astype(jnp.float32), dtype=np.float64)
    actual_count = np.asarray(actual_state.count, dtype=np.int64)
    actual_mean = np.asarray(actual_state.mean, dtype=np.float64)
    actual_var = np.asarray(actual_state.var, dtype=np.float64)
    actual_prefix_var = np.asarray(prefix_variance, dtype=np.float64)
    output_error = _error_stats(actual_output, expected_y)
    mean_error = _error_stats(actual_mean, expected_state.mean)
    var_error = _error_stats(actual_var, expected_state.var)
    tolerances = _case_tolerances(case, expected_state)

    output_close = bool(
        np.allclose(
            actual_output,
            expected_y,
            atol=tolerances["output_atol"],
            rtol=tolerances["output_rtol"],
        )
    )
    count_exact = bool(np.array_equal(actual_count, expected_state.count))
    mean_close = bool(
        np.allclose(
            actual_mean,
            expected_state.mean,
            atol=tolerances["mean_atol"],
            rtol=tolerances["mean_rtol"],
        )
    )
    var_close = bool(
        np.allclose(
            actual_var,
            expected_state.var,
            atol=tolerances["var_atol"],
            rtol=tolerances["var_rtol"],
        )
    )
    masked_exact_zero = bool(np.all(actual_output[~valid] == 0.0))
    nonfinite = {
        "output": int(np.size(actual_output) - np.count_nonzero(np.isfinite(actual_output))),
        "mean": int(np.size(actual_mean) - np.count_nonzero(np.isfinite(actual_mean))),
        "variance": int(np.size(actual_var) - np.count_nonzero(np.isfinite(actual_var))),
        "prefix_variance": int(
            np.size(actual_prefix_var) - np.count_nonzero(np.isfinite(actual_prefix_var))
        ),
    }
    minimum_variance = float(np.min(actual_prefix_var))
    variance_nonnegative = minimum_variance >= -tolerances["var_atol"]
    passed = bool(
        output_close
        and count_exact
        and mean_close
        and var_close
        and masked_exact_zero
        and not any(nonfinite.values())
        and variance_nonnegative
    )
    execution_path = "direct"
    if candidate_name == "candidate_c_shifted_cumsum":
        hot_path = timestep_norm_candidates.candidate_c_hot_path_supported(
            mask,
            segments,
        )
        execution_path = "one_cumsum_hot_path" if hot_path else "candidate_a_fallback"
    return {
        "candidate": candidate_name,
        "name": case.name,
        "family": case.family,
        "dtype": case.dtype,
        "execution_path": execution_path,
        "passed": passed,
        "errors": {
            "output": dataclasses.asdict(output_error),
            "mean": dataclasses.asdict(mean_error),
            "variance": dataclasses.asdict(var_error),
        },
        "invariants": {
            "output_close": output_close,
            "count_exact": count_exact,
            "mean_close": mean_close,
            "variance_close": var_close,
            "masked_outputs_exact_zero": masked_exact_zero,
            "nonfinite": nonfinite,
            "minimum_variance": minimum_variance,
            "variance_nonnegative_within_tolerance": variance_nonnegative,
        },
        "counts": {
            "actual": actual_count,
            "expected": expected_state.count,
        },
        "tolerances": tolerances,
    }


def _make_cases(
    seed: int, *, batch: int, length: int, features: int, groups: int
) -> list[CaseSpec]:
    """Construct deterministic normal and adversarial input families."""
    rng = np.random.default_rng(seed)
    shape = (batch, length, features)
    normal = rng.normal(size=shape)
    feature_wave = np.sin(np.arange(features, dtype=np.float64) * 0.7)[None, None, :]
    time = np.arange(length, dtype=np.float64)[None, :, None]
    batch_shift = np.arange(batch, dtype=np.float64)[:, None, None]

    leading = np.ones((batch, length), dtype=np.bool_)
    leading[0, :5] = False
    leading[1:, :3] = False
    internal = np.ones((batch, length), dtype=np.bool_)
    internal[:, 2::5] = False
    internal[0, 1] = False
    trailing = np.ones((batch, length), dtype=np.bool_)
    trailing[0, -7:] = False
    trailing[1:, -4:] = False
    fully_masked = np.zeros((batch, length), dtype=np.bool_)

    segments = np.zeros((batch, length), dtype=np.int32)
    templates = (
        (0, 1, 1, 1, 0, 2, 2, 2, 2, 0, 1, 1, 1, 0, 3, 3, 3, 3, 3),
        (4, 4, 4, 0, 5, 5, 0, 6, 6, 6, 0, 4, 4, 4, 4, 0, 7, 7),
    )
    for row in range(batch):
        template = templates[row % len(templates)]
        segments[row, : min(length, len(template))] = template[:length]
    packed_mask = np.ones((batch, length), dtype=np.bool_)
    packed_mask[:, 6::9] = False

    prior_mean = np.linspace(-2.0, 3.0, groups, dtype=np.float64)
    prior_logv = np.log(np.linspace(0.25, 2.0, groups, dtype=np.float64))
    continuation = OracleState(
        count=np.arange(batch, dtype=np.int64) + 7,
        mean=np.stack([np.linspace(8.0 + row, 12.0 + row, groups) for row in range(batch)]),
        var=np.stack(
            [np.linspace(0.2 + 0.1 * row, 1.4 + 0.1 * row, groups) for row in range(batch)]
        ),
    )

    cases = [
        CaseSpec("normal", "normal", normal),
        CaseSpec("nonzero_mean", "nonzero_mean", normal * 1.7 + 37.0),
        CaseSpec("constant", "constant", np.full(shape, 7.25)),
        CaseSpec("almost_constant", "almost_constant", 3.0 + normal * 1e-4, output_atol=1e-2),
        CaseSpec(
            "alternating",
            "alternating",
            np.broadcast_to(
                ((-1.0) ** (time + np.arange(features)[None, None, :]))
                * (0.5 + np.abs(feature_wave)),
                shape,
            ).copy(),
        ),
        CaseSpec("drift", "drift", 0.075 * time + 0.5 * batch_shift + feature_wave + normal * 0.02),
    ]
    for offset in (1e2, 1e4, 1e6):
        cases.append(
            CaseSpec(
                f"offset_{offset:.0e}_small_variance",
                f"offset_{offset:.0e}",
                offset + normal * (offset * 2e-6),
                output_atol=0.2,
                output_rtol=0.03,
            )
        )
    cases.extend(
        [
            CaseSpec(
                "bfloat16_nonzero_mean",
                "bfloat16",
                normal * 2.0 + 11.0,
                dtype="bfloat16",
                output_atol=2e-2,
                output_rtol=2e-2,
            ),
            CaseSpec(
                "learned_prior",
                "prior",
                normal + 5.0,
                prior_count=5,
                prior_mean=prior_mean,
                prior_logv=prior_logv,
            ),
            CaseSpec(
                "continuation",
                "continuation",
                normal + 4.0,
                state=continuation,
                mask=internal,
            ),
            CaseSpec("leading_mask", "mask_leading", normal + 2.0, mask=leading),
            CaseSpec("internal_mask", "mask_internal", normal - 1.0, mask=internal),
            CaseSpec("trailing_mask", "mask_trailing", normal + 0.5, mask=trailing),
            CaseSpec("fully_masked", "fully_masked", normal + 9.0, mask=fully_masked),
            CaseSpec(
                "packed_resets",
                "packed",
                normal + 6.0,
                prior_count=3,
                prior_mean=prior_mean,
                prior_logv=prior_logv,
                mask=packed_mask,
                segment_ids=segments,
            ),
        ]
    )
    return cases


def _oracle_objective(
    values: np.ndarray,
    *,
    groups: int,
    norm: TimestepNorm,
    weight: np.ndarray,
    bias: np.ndarray,
    prior_mean: np.ndarray | None,
    prior_logv: np.ndarray | None,
    state: OracleState | None,
    mask: np.ndarray | None,
    segment_ids: np.ndarray | None,
    output_coefficients: np.ndarray,
    mean_coefficients: np.ndarray,
    variance_coefficients: np.ndarray,
) -> float:
    """Return a smooth scalar objective from the independent oracle."""
    output, final, _ = numpy_timestep_norm(
        values,
        groups=groups,
        eps=norm.eps,
        weight=weight,
        bias=bias,
        prior_count=norm.prior_count,
        prior_mean=prior_mean,
        prior_logv=prior_logv,
        state=state,
        mask=mask,
        segment_ids=segment_ids,
    )
    return float(
        np.sum(output * output_coefficients)
        + 0.1 * np.sum(final.mean * mean_coefficients)
        + 0.03 * np.sum(final.var * variance_coefficients)
    )


def _finite_difference(
    objective: Callable[[], float],
    target: np.ndarray,
    *,
    step: float,
) -> np.ndarray:
    """Compute central finite differences by mutating a private float64 array."""
    gradient = np.empty_like(target, dtype=np.float64)
    for index in np.ndindex(target.shape):
        original = float(target[index])
        target[index] = original + step
        positive = objective()
        target[index] = original - step
        negative = objective()
        target[index] = original
        gradient[index] = (positive - negative) / (2.0 * step)
    return gradient


def _gradient_cases(seed: int) -> list[CaseSpec]:
    """Return small smooth cases suitable for independent finite differences."""
    rng = np.random.default_rng(seed + 991)
    values = rng.normal(size=(1, 5, 4)).astype(np.float64) + 1.5
    mask = np.asarray([[True, False, True, True, True]])
    prior_mean = np.asarray([-0.5, 0.75], dtype=np.float64)
    prior_logv = np.log(np.asarray([0.4, 1.6], dtype=np.float64))
    continuation = OracleState(
        count=np.asarray([4], dtype=np.int64),
        mean=np.asarray([[2.0, -1.0]], dtype=np.float64),
        var=np.asarray([[0.7, 1.3]], dtype=np.float64),
    )
    return [
        CaseSpec("gradient_plain", "gradient_plain", values),
        CaseSpec("gradient_masked", "gradient_masked", values, mask=mask),
        CaseSpec(
            "gradient_prior",
            "gradient_prior",
            values,
            prior_count=3,
            prior_mean=prior_mean,
            prior_logv=prior_logv,
        ),
        CaseSpec(
            "gradient_continuation",
            "gradient_continuation",
            values,
            state=continuation,
        ),
    ]


def _evaluate_gradient_case(
    candidate: Candidate,
    case: CaseSpec,
    *,
    groups: int,
    seed: int,
) -> dict[str, Any]:
    """Compare JAX reverse-mode gradients with NumPy central differences."""
    features = case.values.shape[-1]
    norm = _make_norm(case, features, groups)
    x = jnp.asarray(case.values, dtype=jnp.float32)
    state = _jax_state(case.state)
    mask = None if case.mask is None else jnp.asarray(case.mask, dtype=jnp.bool_)
    segments = None if case.segment_ids is None else jnp.asarray(case.segment_ids, dtype=jnp.int32)
    rng = np.random.default_rng(seed + 173)
    output_coefficients = rng.normal(size=case.values.shape).astype(np.float64)
    mean_coefficients = rng.normal(size=(case.values.shape[0], groups)).astype(np.float64)
    variance_coefficients = rng.normal(size=(case.values.shape[0], groups)).astype(np.float64)
    output_coefficients_jax = jnp.asarray(output_coefficients, dtype=jnp.float32)
    mean_coefficients_jax = jnp.asarray(mean_coefficients, dtype=jnp.float32)
    variance_coefficients_jax = jnp.asarray(variance_coefficients, dtype=jnp.float32)

    def objective_value(
        module: TimestepNorm,
        values: Array,
        incoming: NormState | None,
    ) -> Array:
        output, final = candidate(module, values, incoming, mask, segments)
        return (
            jnp.sum(output.astype(jnp.float32) * output_coefficients_jax)
            + 0.1 * jnp.sum(final.mean * mean_coefficients_jax)
            + 0.03 * jnp.sum(final.var * variance_coefficients_jax)
        )

    state_mean_actual = None
    state_var_actual = None
    if state is None:
        _, (norm_gradient, input_gradient) = jax.value_and_grad(
            lambda module, values: objective_value(module, values, None),
            argnums=(0, 1),
        )(norm, x)
    else:

        def objective_with_state(
            module: TimestepNorm,
            values: Array,
            incoming_mean: Array,
            incoming_var: Array,
        ) -> Array:
            incoming = NormState(
                count=state.count,
                mean=incoming_mean,
                var=incoming_var,
            )
            return objective_value(module, values, incoming)

        _, gradients = jax.value_and_grad(
            objective_with_state,
            argnums=(0, 1, 2, 3),
        )(norm, x, state.mean, state.var)
        norm_gradient, input_gradient, state_mean_gradient, state_var_gradient = gradients
        state_mean_actual = np.asarray(state_mean_gradient, dtype=np.float64)
        state_var_actual = np.asarray(state_var_gradient, dtype=np.float64)
    input_actual = np.asarray(input_gradient, dtype=np.float64)
    weight_actual = np.asarray(norm_gradient.weight, dtype=np.float64)
    bias_actual = np.asarray(norm_gradient.bias, dtype=np.float64)
    prior_mean_actual = (
        None if norm_gradient.prior_mean is None else np.asarray(norm_gradient.prior_mean)
    )
    prior_logv_actual = (
        None if norm_gradient.prior_logv is None else np.asarray(norm_gradient.prior_logv)
    )

    values64 = np.asarray(case.values, dtype=np.float64).copy()
    weight64 = np.asarray(norm.weight, dtype=np.float64).copy()
    bias64 = np.asarray(norm.bias, dtype=np.float64).copy()
    prior_mean64 = (
        None if norm.prior_mean is None else np.asarray(norm.prior_mean, dtype=np.float64).copy()
    )
    prior_logv64 = (
        None if norm.prior_logv is None else np.asarray(norm.prior_logv, dtype=np.float64).copy()
    )
    state_mean64 = (
        None if case.state is None else np.asarray(case.state.mean, dtype=np.float64).copy()
    )
    state_var64 = (
        None if case.state is None else np.asarray(case.state.var, dtype=np.float64).copy()
    )

    def oracle_objective() -> float:
        oracle_state = None
        if case.state is not None:
            assert state_mean64 is not None and state_var64 is not None
            oracle_state = OracleState(
                count=case.state.count,
                mean=state_mean64,
                var=state_var64,
            )
        return _oracle_objective(
            values64,
            groups=groups,
            norm=norm,
            weight=weight64,
            bias=bias64,
            prior_mean=prior_mean64,
            prior_logv=prior_logv64,
            state=oracle_state,
            mask=case.mask,
            segment_ids=case.segment_ids,
            output_coefficients=output_coefficients,
            mean_coefficients=mean_coefficients,
            variance_coefficients=variance_coefficients,
        )

    expected: dict[str, np.ndarray] = {
        "input": _finite_difference(oracle_objective, values64, step=2e-4),
        "weight": _finite_difference(oracle_objective, weight64, step=2e-4),
        "bias": _finite_difference(oracle_objective, bias64, step=2e-4),
    }
    actual: dict[str, np.ndarray] = {
        "input": input_actual,
        "weight": weight_actual,
        "bias": bias_actual,
    }
    if prior_mean64 is not None and prior_mean_actual is not None:
        expected["prior_mean"] = _finite_difference(oracle_objective, prior_mean64, step=2e-4)
        actual["prior_mean"] = np.asarray(prior_mean_actual, dtype=np.float64)
    if prior_logv64 is not None and prior_logv_actual is not None:
        expected["prior_logv"] = _finite_difference(oracle_objective, prior_logv64, step=2e-4)
        actual["prior_logv"] = np.asarray(prior_logv_actual, dtype=np.float64)
    if state_mean64 is not None and state_mean_actual is not None:
        expected["state_mean"] = _finite_difference(oracle_objective, state_mean64, step=2e-4)
        actual["state_mean"] = state_mean_actual
    if state_var64 is not None and state_var_actual is not None:
        expected["state_var"] = _finite_difference(oracle_objective, state_var64, step=2e-4)
        actual["state_var"] = state_var_actual

    leaves: dict[str, Any] = {}
    passed = True
    for name in expected:
        error = _error_stats(actual[name], expected[name])
        close = bool(np.allclose(actual[name], expected[name], atol=2e-3, rtol=8e-3))
        finite = bool(np.all(np.isfinite(actual[name])))
        leaves[name] = {
            "passed": close and finite,
            "finite": finite,
            "errors": dataclasses.asdict(error),
        }
        passed &= close and finite
    return {"name": case.name, "family": case.family, "passed": passed, "leaves": leaves}


def _aggregate_families(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate maximum numerical errors and invariants by case family."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        grouped.setdefault(case["family"], []).append(case)
    result: dict[str, Any] = {}
    for family, members in grouped.items():

        def maximum_error(quantity: str, kind: str) -> float | None:
            values = [member["errors"][quantity][kind] for member in members]
            return None if any(value is None for value in values) else max(values)

        result[family] = {
            "case_count": len(members),
            "passed": all(member["passed"] for member in members),
            "max_abs_error": {
                quantity: maximum_error(quantity, "max_abs")
                for quantity in ("output", "mean", "variance")
            },
            "max_relative_error": {
                quantity: maximum_error(quantity, "max_relative")
                for quantity in ("output", "mean", "variance")
            },
            "counts_exact": all(member["invariants"]["count_exact"] for member in members),
            "masked_outputs_exact_zero": all(
                member["invariants"]["masked_outputs_exact_zero"] for member in members
            ),
            "nonfinite": {
                quantity: sum(member["invariants"]["nonfinite"][quantity] for member in members)
                for quantity in ("output", "mean", "variance", "prefix_variance")
            },
            "minimum_variance": (
                min(
                    member["invariants"]["minimum_variance"]
                    for member in members
                    if member["invariants"]["minimum_variance"] is not None
                )
                if any(member["invariants"]["minimum_variance"] is not None for member in members)
                else None
            ),
        }
    return result


def _forward_exception(candidate_name: str, case: CaseSpec, exc: Exception) -> dict[str, Any]:
    """Represent a candidate runtime failure without losing the JSON report."""
    empty_error = {"max_abs": None, "max_relative": None}
    return {
        "candidate": candidate_name,
        "name": case.name,
        "family": case.family,
        "dtype": case.dtype,
        "execution_path": "unavailable",
        "passed": False,
        "exception": {"type": type(exc).__name__, "message": str(exc)},
        "errors": {
            "output": dict(empty_error),
            "mean": dict(empty_error),
            "variance": dict(empty_error),
        },
        "invariants": {
            "output_close": False,
            "count_exact": False,
            "mean_close": False,
            "variance_close": False,
            "masked_outputs_exact_zero": False,
            "nonfinite": {
                "output": 0,
                "mean": 0,
                "variance": 0,
                "prefix_variance": 0,
            },
            "minimum_variance": None,
            "variance_nonnegative_within_tolerance": False,
        },
        "counts": {"actual": None, "expected": None},
        "tolerances": None,
    }


def _gradient_exception(case: CaseSpec, exc: Exception) -> dict[str, Any]:
    """Represent a gradient runtime failure in the machine-readable report."""
    return {
        "name": case.name,
        "family": case.family,
        "passed": False,
        "exception": {"type": type(exc).__name__, "message": str(exc)},
        "leaves": {},
    }


def _candidate_registry() -> dict[str, Candidate]:
    """Load the candidate mapping, with a fail-loud compatibility fallback."""
    registry = getattr(timestep_norm_candidates, "CANDIDATES", None)
    if registry is None:
        aliases = {
            "associative": ("associative_timestep_norm", "hillis_steele_timestep_norm"),
            "shifted_cumsum": ("shifted_cumsum_timestep_norm",),
            "serial_stats_only": (
                "serial_stats_only_timestep_norm",
                "serial_stats_timestep_norm",
            ),
        }
        registry = {}
        for canonical, names in aliases.items():
            for name in names:
                if hasattr(timestep_norm_candidates, name):
                    registry[canonical] = getattr(timestep_norm_candidates, name)
                    break
    if not isinstance(registry, Mapping) or not registry:
        raise RuntimeError("tools.timestep_norm_candidates must expose a nonempty CANDIDATES map")
    invalid = [name for name, candidate in registry.items() if not callable(candidate)]
    if invalid:
        raise TypeError(f"non-callable candidate entries: {invalid}")
    candidates = {str(name): candidate for name, candidate in registry.items()}
    candidates["production"] = _production_timestep_norm
    return candidates


def main() -> int:
    """Run all selected candidates and write machine-readable results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "local-scratch" / "timestep-norm-candidate-verification.json",
    )
    parser.add_argument("--candidate", action="append", dest="candidates", default=[])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=37)
    parser.add_argument("--features", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    args = parser.parse_args()
    if args.features <= 0 or args.groups <= 0 or args.features % args.groups:
        parser.error("--features must be positive and divisible by --groups")
    if args.batch <= 0 or args.length <= 0:
        parser.error("--batch and --length must be positive")

    registry = _candidate_registry()
    selected = args.candidates or sorted(registry)
    unknown = sorted(set(selected) - set(registry))
    if unknown:
        parser.error(f"unknown candidates {unknown}; choices are {sorted(registry)}")

    cases = _make_cases(
        args.seed,
        batch=args.batch,
        length=args.length,
        features=args.features,
        groups=args.groups,
    )
    gradient_cases = _gradient_cases(args.seed)
    candidate_results: dict[str, Any] = {}
    failed_candidates: list[str] = []
    for name in selected:
        candidate = registry[name]
        forward_results = []
        for case in cases:
            try:
                result = _evaluate_case(name, candidate, case, groups=args.groups)
            except Exception as exc:  # noqa: BLE001 - candidates are the subject under test.
                result = _forward_exception(name, case, exc)
            forward_results.append(result)
        gradient_results = []
        for index, case in enumerate(gradient_cases):
            try:
                result = _evaluate_gradient_case(
                    candidate,
                    case,
                    groups=2,
                    seed=args.seed + index,
                )
            except Exception as exc:  # noqa: BLE001 - preserve failures in the report.
                result = _gradient_exception(case, exc)
            gradient_results.append(result)
        passed = all(case["passed"] for case in forward_results) and all(
            case["passed"] for case in gradient_results
        )
        if not passed:
            failed_candidates.append(name)
        candidate_results[name] = {
            "passed": passed,
            "forward_cases": forward_results,
            "families": _aggregate_families(forward_results),
            "gradient_cases": gradient_results,
            "summary": {
                "forward_passed": sum(case["passed"] for case in forward_results),
                "forward_total": len(forward_results),
                "gradient_passed": sum(case["passed"] for case in gradient_results),
                "gradient_total": len(gradient_results),
            },
        }

    payload = {
        "schema_version": 1,
        "repository": str(args.repo_root.resolve()),
        "repository_commit": _git_revision(args.repo_root.resolve()),
        "environment": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "equinox": eqx.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
        },
        "configuration": {
            "seed": args.seed,
            "batch": args.batch,
            "length": args.length,
            "features": args.features,
            "groups": args.groups,
            "selected_candidates": selected,
        },
        "candidates": candidate_results,
        "summary": {
            "passed": not failed_candidates,
            "candidate_passed": len(selected) - len(failed_candidates),
            "candidate_total": len(selected),
            "failed_candidates": failed_candidates,
        },
    }
    rendered = json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")

    print(f"TimestepNorm candidate verification ({jax.default_backend()})")
    for name in selected:
        result = candidate_results[name]
        summary = result["summary"]
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status:4s} {name:24s} "
            f"forward {summary['forward_passed']}/{summary['forward_total']}, "
            f"gradients {summary['gradient_passed']}/{summary['gradient_total']}"
        )
    print(f"JSON: {args.output}")
    if failed_candidates:
        print(f"Failed candidates: {', '.join(failed_candidates)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
