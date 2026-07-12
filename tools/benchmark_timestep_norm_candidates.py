#!/usr/bin/env python3
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
"""Benchmark pure-JAX TimestepNorm prefix candidates on a GPU.

The driver compares production TimestepNorm, Candidates A/B/C, and the compact
serial statistics oracle. Every measurement lowers and compiles an outermost
JIT, warms a device-resident input, synchronizes every timed invocation, and
reports forward plus forward-and-backward latency. Backward-only latency is the
difference between those independently synchronized distributions.

Default matrix:

- batch: 1, 2, 4
- sequence: 64, 128, 256, 512, 1024, 2048, 4096
- feature/group: 1024/32
- dtype: FP32, BF16
- mode: plain, masked, packed, continuation

Candidate C's one-cumsum hot path supports only plain and continuation modes.
Masked and packed calls execute Candidate A as an exact semantic fallback and
are labeled accordingly in both console and JSON output.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from timestep_norm_candidates import CANDIDATES, candidate_c_hot_path_supported

from megalodon_jax.layers.timestep_norm import TimestepNorm
from megalodon_jax.types import NormState

Mode = Literal["plain", "masked", "packed", "continuation"]
Variant = Callable[
    [TimestepNorm, jax.Array, NormState | None, jax.Array | None, jax.Array | None],
    tuple[jax.Array, NormState],
]

DEFAULT_BATCH_SIZES = (1, 2, 4)
DEFAULT_SEQUENCE_LENGTHS = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_FEATURE_DIMS = (1024,)
DEFAULT_GROUP_COUNTS = (32,)
DEFAULT_DTYPES = ("float32", "bfloat16")
DEFAULT_MODES: tuple[Mode, ...] = ("plain", "masked", "packed", "continuation")
DEFAULT_VARIANTS = (
    "production",
    "serial_stats_only_oracle",
    "candidate_a_associative",
    "candidate_b_hillis_steele",
    "candidate_c_shifted_cumsum",
)


class ModeInputs(NamedTuple):
    """Optional state and masks associated with one execution mode."""

    state: NormState | None
    mask: jax.Array | None
    segment_ids: jax.Array | None


def production_timestep_norm(
    norm: TimestepNorm,
    x: jax.Array,
    state: NormState | None,
    mask: jax.Array | None,
    segment_ids: jax.Array | None,
) -> tuple[jax.Array, NormState]:
    """Call the production layer through the candidate-compatible signature."""
    return norm(x, state=state, mask=mask, segment_ids=segment_ids)


VARIANTS: dict[str, Variant] = {
    "production": production_timestep_norm,
    **CANDIDATES,
}


def _block_tree(tree: Any) -> None:
    """Synchronize every device array in a pytree."""
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _device_put_tree(tree: Any) -> Any:
    """Place every array leaf on the default device without changing metadata."""
    placed = jax.tree.map(
        lambda leaf: jax.device_put(leaf) if eqx.is_array(leaf) else leaf,
        tree,
    )
    _block_tree(placed)
    return placed


def _percentile(samples: Sequence[float], q: float) -> float:
    """Return a percentile as a JSON-native float."""
    return float(np.percentile(np.asarray(samples, dtype=np.float64), q))


def _timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    """Summarize synchronized wall-clock samples."""
    return {
        "iterations": len(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p90_ms": _percentile(samples_ms, 90),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def _measure(
    compiled: Any,
    call_args: tuple[Any, ...],
    *,
    warmups: int,
    iterations: int,
) -> tuple[dict[str, Any], Any]:
    """Warm and measure a compiled executable with explicit synchronization."""
    for _ in range(warmups):
        _block_tree(compiled(*call_args))

    samples: list[float] = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = compiled(*call_args)
        _block_tree(result)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return _timing_summary(samples), result


def _stablehlo_metadata(lowered: Any) -> tuple[dict[str, Any], str | None]:
    """Return StableHLO size/control-flow metadata when lowering exposes it."""
    try:
        text = str(lowered.compiler_ir(dialect="stablehlo"))
    except Exception as error:  # pragma: no cover - backend/version dependent
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }, None
    return {
        "available": True,
        "bytes": len(text.encode("utf-8")),
        "while_count": text.count("stablehlo.while"),
        "dynamic_update_slice_count": text.count("stablehlo.dynamic_update_slice"),
        "reduce_window_count": text.count("stablehlo.reduce_window"),
        "custom_call_count": text.count("stablehlo.custom_call"),
    }, text


def _public_attributes(value: Any) -> dict[str, Any]:
    """Extract scalar public fields from a backend analysis object."""
    result: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:  # pragma: no cover - defensive reflection
            continue
        if callable(item):
            continue
        if item is None or isinstance(item, (bool, int, float, str, np.generic)):
            result[name] = _jsonable(item)
    return result


def _memory_analysis(compiled: Any) -> dict[str, Any]:
    """Return compiler memory estimates when the backend provides them."""
    try:
        analysis = compiled.memory_analysis()
    except Exception as error:  # pragma: no cover - backend/version dependent
        return {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }
    if analysis is None:
        return {"available": False, "error": "backend returned no memory analysis"}
    fields = _public_attributes(analysis)
    return {"available": True, **fields}


def _cost_analysis(compiled: Any, lowered: Any) -> dict[str, Any]:
    """Return compiled or lowered cost estimates when available."""
    errors: list[str] = []
    for source, candidate in (("compiled", compiled), ("lowered", lowered)):
        method = getattr(candidate, "cost_analysis", None)
        if method is None:
            continue
        try:
            analysis = method()
        except Exception as error:  # pragma: no cover - backend/version dependent
            errors.append(f"{source}: {type(error).__name__}: {error}")
            continue
        if analysis is not None:
            return {
                "available": True,
                "source": source,
                "values": _jsonable(analysis),
            }
    return {
        "available": False,
        "errors": errors or ["cost_analysis unavailable"],
    }


def _compile(
    jitted: Any,
    call_args: tuple[Any, ...],
    *,
    hlo_path: Path | None,
) -> tuple[Any, dict[str, Any]]:
    """Lower and compile one outermost JIT while collecting compiler metadata."""
    lower_start = time.perf_counter()
    lowered = jitted.lower(*call_args)
    lower_seconds = time.perf_counter() - lower_start
    hlo_metadata, hlo_text = _stablehlo_metadata(lowered)
    if hlo_path is not None and hlo_text is not None:
        hlo_path.parent.mkdir(parents=True, exist_ok=True)
        hlo_path.write_text(hlo_text)

    compile_start = time.perf_counter()
    compiled = lowered.compile()
    compile_seconds = time.perf_counter() - compile_start
    metadata = {
        "lower_seconds": lower_seconds,
        "compile_seconds": compile_seconds,
        "total_lower_compile_seconds": lower_seconds + compile_seconds,
        "stablehlo": hlo_metadata,
        "compiled_memory_analysis": _memory_analysis(compiled),
        "cost_analysis": _cost_analysis(compiled, lowered),
    }
    return compiled, metadata


def _make_loss(variant: Variant) -> Callable[..., jax.Array]:
    """Create a scalar loss that keeps output and continuation statistics live."""

    def loss(
        norm: TimestepNorm,
        x: jax.Array,
        state: NormState | None,
        mask: jax.Array | None,
        segment_ids: jax.Array | None,
    ) -> jax.Array:
        output, final_state = variant(norm, x, state, mask, segment_ids)
        output_f32 = output.astype(jnp.float32)
        return (
            jnp.mean(jnp.sin(output_f32) + 0.01 * jnp.square(output_f32))
            + 1e-3 * jnp.mean(final_state.mean)
            + 1e-4 * jnp.mean(final_state.var)
        )

    return loss


def _compile_and_measure_variant(
    name: str,
    variant: Variant,
    call_args: tuple[Any, ...],
    *,
    mode_inputs: ModeInputs,
    warmups: int,
    iterations: int,
    hlo_dir: Path | None,
    hlo_stem: str,
) -> tuple[dict[str, Any], Any]:
    """Compile and measure forward and forward+backward for one variant."""
    forward_jit = jax.jit(variant)
    loss = _make_loss(variant)
    forward_backward_jit = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))

    forward_path = None if hlo_dir is None else hlo_dir / f"{hlo_stem}_{name}_forward.mlir"
    backward_path = (
        None if hlo_dir is None else hlo_dir / f"{hlo_stem}_{name}_forward_backward.mlir"
    )
    forward_compiled, forward_compiler = _compile(
        forward_jit,
        call_args,
        hlo_path=forward_path,
    )
    forward_backward_compiled, forward_backward_compiler = _compile(
        forward_backward_jit,
        call_args,
        hlo_path=backward_path,
    )
    forward_timing, forward_output = _measure(
        forward_compiled,
        call_args,
        warmups=warmups,
        iterations=iterations,
    )
    forward_backward_timing, _ = _measure(
        forward_backward_compiled,
        call_args,
        warmups=warmups,
        iterations=iterations,
    )

    c_hot_path = candidate_c_hot_path_supported(
        mode_inputs.mask,
        mode_inputs.segment_ids,
    )
    if name == "candidate_c_shifted_cumsum":
        execution_path = "one_cumsum_hot_path" if c_hot_path else "fallback_candidate_a_associative"
    elif name == "serial_stats_only_oracle":
        execution_path = "compact_serial_oracle"
    elif name == "production":
        execution_path = "production"
    else:
        execution_path = "candidate"

    backward_only = {
        "method": "forward_plus_backward minus forward (separate synchronized samples)",
        "median_ms": forward_backward_timing["median_ms"] - forward_timing["median_ms"],
        "p90_ms": forward_backward_timing["p90_ms"] - forward_timing["p90_ms"],
    }
    result = {
        "status": "passed",
        "execution_path": execution_path,
        "candidate_c_fallback": name == "candidate_c_shifted_cumsum" and not c_hot_path,
        "forward": {
            "timing": forward_timing,
            "compiler": forward_compiler,
        },
        "forward_backward": {
            "timing": forward_backward_timing,
            "compiler": forward_backward_compiler,
        },
        "derived_backward_only": backward_only,
    }
    return result, forward_output


def _make_mask(batch: int, length: int) -> jax.Array:
    """Create a deterministic mask containing internal and trailing invalid tokens."""
    positions = jnp.arange(length, dtype=jnp.int32)[None]
    rows = jnp.arange(batch, dtype=jnp.int32)[:, None]
    mask = ((positions + 2 * rows) % 7 != 2) & ((positions + rows) % 11 != 5)
    return mask.at[:, -1].set(False) if length else mask


def _make_segments(batch: int, length: int) -> jax.Array:
    """Create deterministic contiguous packed runs separated by ID-zero padding."""
    values = np.zeros((batch, length), dtype=np.int32)
    for row in range(batch):
        cursor = 0
        segment = 1
        while cursor < length:
            width = min(3 + ((row + segment) % 6), length - cursor)
            values[row, cursor : cursor + width] = segment
            cursor += width
            if cursor < length:
                cursor += 1
            segment += 1
    return jnp.asarray(values)


def _make_mode_inputs(
    mode: Mode,
    *,
    batch: int,
    length: int,
    groups: int,
    key: jax.Array,
) -> ModeInputs:
    """Create deterministic optional inputs for one benchmark mode."""
    if mode == "plain":
        return ModeInputs(None, None, None)
    if mode == "masked":
        return ModeInputs(None, _make_mask(batch, length), None)
    if mode == "packed":
        return ModeInputs(None, None, _make_segments(batch, length))
    if mode == "continuation":
        mean_key, var_key = jax.random.split(key)
        state = NormState(
            count=jnp.arange(batch, dtype=jnp.int32) + 3,
            mean=jax.random.normal(mean_key, (batch, groups), dtype=jnp.float32),
            var=jnp.exp(jax.random.normal(var_key, (batch, groups), dtype=jnp.float32)),
        )
        return ModeInputs(state, None, None)
    raise ValueError(f"unknown mode: {mode}")


def _max_abs(left: jax.Array, right: jax.Array) -> float:
    """Return synchronized maximum absolute FP32 error."""
    return float(jnp.max(jnp.abs(left.astype(jnp.float32) - right.astype(jnp.float32))))


def _correctness(actual: Any, expected: Any) -> dict[str, Any]:
    """Compare one forward result against the serial statistics oracle."""
    actual_y, actual_state = actual
    expected_y, expected_state = expected
    return {
        "output_max_abs": _max_abs(actual_y, expected_y),
        "count_equal": bool(
            np.array_equal(np.asarray(actual_state.count), np.asarray(expected_state.count))
        ),
        "mean_max_abs": _max_abs(actual_state.mean, expected_state.mean),
        "var_max_abs": _max_abs(actual_state.var, expected_state.var),
    }


def _jsonable(value: Any) -> Any:
    """Recursively convert compiler metadata to JSON-native values."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # pragma: no cover - defensive conversion
            pass
    return str(value)


def _case_line(case: Mapping[str, Any]) -> str:
    """Render one concise human-readable case summary."""
    prefix = (
        f"B={case['batch']} L={case['length']} D={case['features']} "
        f"G={case['groups']} {case['dtype']} {case['mode']}"
    )
    parts: list[str] = []
    for name, result in case["variants"].items():
        if result["status"] != "passed":
            parts.append(f"{name}=ERROR")
            continue
        path = "[fallback:A]" if result["candidate_c_fallback"] else ""
        forward = result["forward"]["timing"]["median_ms"]
        forward_backward = result["forward_backward"]["timing"]["median_ms"]
        parts.append(f"{name}{path}={forward:.3f}/{forward_backward:.3f}ms")
    return prefix + " | " + " ".join(parts)


def _validate_args(args: argparse.Namespace) -> None:
    """Reject invalid grids before allocating or compiling anything."""
    for name in ("batch_sizes", "sequence_lengths", "feature_dims", "group_counts"):
        values = getattr(args, name)
        if not values or min(values) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} values must be positive")
    invalid_pairs = [
        (features, groups)
        for features in args.feature_dims
        for groups in args.group_counts
        if features % groups
    ]
    if invalid_pairs:
        raise SystemExit(f"feature dimensions must be divisible by groups: {invalid_pairs}")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if args.prior_count < 0:
        raise SystemExit("--prior-count must be non-negative")


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument(
        "--sequence-lengths",
        "--sequences",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEQUENCE_LENGTHS),
    )
    parser.add_argument("--feature-dims", nargs="+", type=int, default=list(DEFAULT_FEATURE_DIMS))
    parser.add_argument("--group-counts", nargs="+", type=int, default=list(DEFAULT_GROUP_COUNTS))
    parser.add_argument(
        "--dtypes",
        nargs="+",
        choices=DEFAULT_DTYPES,
        default=list(DEFAULT_DTYPES),
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=DEFAULT_MODES,
        default=list(DEFAULT_MODES),
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(VARIANTS),
        default=list(DEFAULT_VARIANTS),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--prior-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-hlo-dir", type=Path)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--allow-non-gpu", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main() -> int:
    """Run the configured benchmark matrix and save machine-readable results."""
    args = _parser().parse_args()
    _validate_args(args)
    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_non_gpu:
        raise SystemExit(
            f"GPU benchmark requires the JAX GPU backend, got {backend!r}; "
            "use --allow-non-gpu only for driver smoke tests"
        )

    measurement_standard = args.warmups >= 3 and args.iterations >= 20
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "equinox": eqx.__version__,
            "backend": backend,
            "devices": [str(device) for device in jax.devices()],
            "xla_flags": os.environ.get("XLA_FLAGS"),
            "jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
            "command": sys.argv,
        },
        "configuration": {
            "batch_sizes": args.batch_sizes,
            "sequence_lengths": args.sequence_lengths,
            "feature_dims": args.feature_dims,
            "group_counts": args.group_counts,
            "dtypes": args.dtypes,
            "modes": args.modes,
            "variants": args.variants,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "prior_count": args.prior_count,
            "seed": args.seed,
            "measurement_standard_met": measurement_standard,
            "candidate_c_contract": {
                "hot_path_modes": ["plain", "continuation"],
                "fallback_modes": ["masked", "packed"],
                "fallback": "candidate_a_associative",
            },
        },
        "cases": [],
    }

    failures = 0
    measurements = 0
    fallback_measurements = 0
    case_index = 0
    profile_started = False
    if args.profile_dir is not None:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(str(args.profile_dir))
        profile_started = True

    try:
        for features in args.feature_dims:
            for groups in args.group_counts:
                for dtype_name in args.dtypes:
                    dtype = jnp.float32 if dtype_name == "float32" else jnp.bfloat16
                    for batch in args.batch_sizes:
                        for length in args.sequence_lengths:
                            for mode_name in args.modes:
                                mode = cast(Mode, mode_name)
                                case_seed = args.seed + case_index * 17
                                case_index += 1
                                key = jax.random.PRNGKey(case_seed)
                                input_key, mode_key = jax.random.split(key)
                                x = (
                                    jax.random.normal(
                                        input_key,
                                        (batch, length, features),
                                        dtype=jnp.float32,
                                    )
                                    * 3.0
                                    + 25.0
                                ).astype(dtype)
                                norm = TimestepNorm(
                                    features,
                                    groups,
                                    prior_count=args.prior_count,
                                )
                                mode_inputs = _make_mode_inputs(
                                    mode,
                                    batch=batch,
                                    length=length,
                                    groups=groups,
                                    key=mode_key,
                                )
                                norm, x, mode_inputs = _device_put_tree((norm, x, mode_inputs))
                                call_args = (
                                    norm,
                                    x,
                                    mode_inputs.state,
                                    mode_inputs.mask,
                                    mode_inputs.segment_ids,
                                )
                                case: dict[str, Any] = {
                                    "case_index": case_index - 1,
                                    "seed": case_seed,
                                    "batch": batch,
                                    "length": length,
                                    "features": features,
                                    "groups": groups,
                                    "dtype": dtype_name,
                                    "mode": mode,
                                    "variants": {},
                                }
                                outputs: dict[str, Any] = {}
                                stem = (
                                    f"b{batch}_l{length}_d{features}_g{groups}_{dtype_name}_{mode}"
                                )
                                for name in args.variants:
                                    try:
                                        result, output = _compile_and_measure_variant(
                                            name,
                                            VARIANTS[name],
                                            call_args,
                                            mode_inputs=mode_inputs,
                                            warmups=args.warmups,
                                            iterations=args.iterations,
                                            hlo_dir=args.dump_hlo_dir,
                                            hlo_stem=stem,
                                        )
                                        case["variants"][name] = result
                                        outputs[name] = output
                                        measurements += 1
                                        fallback_measurements += int(result["candidate_c_fallback"])
                                    except Exception as error:
                                        failures += 1
                                        case["variants"][name] = {
                                            "status": "failed",
                                            "error_type": type(error).__name__,
                                            "error": str(error),
                                        }
                                        if args.fail_fast:
                                            raise

                                oracle = outputs.get("serial_stats_only_oracle")
                                if oracle is not None:
                                    for name, output in outputs.items():
                                        case["variants"][name]["correctness_vs_serial_oracle"] = (
                                            _correctness(output, oracle)
                                        )
                                successful = {
                                    name: result
                                    for name, result in case["variants"].items()
                                    if result["status"] == "passed"
                                }
                                if successful:
                                    case["fastest_forward"] = min(
                                        successful,
                                        key=lambda name: successful[name]["forward"]["timing"][
                                            "median_ms"
                                        ],
                                    )
                                    case["fastest_forward_backward"] = min(
                                        successful,
                                        key=lambda name: successful[name]["forward_backward"][
                                            "timing"
                                        ]["median_ms"],
                                    )
                                payload["cases"].append(case)
                                print(_case_line(case), flush=True)
    finally:
        if profile_started:
            jax.profiler.stop_trace()

    payload["summary"] = {
        "case_count": len(payload["cases"]),
        "measurement_count": measurements,
        "failure_count": failures,
        "candidate_c_fallback_measurements": fallback_measurements,
        "measurement_standard_met": measurement_standard,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")
    print(
        f"Completed {measurements} measurements across {len(payload['cases'])} cases; "
        f"failures={failures}, Candidate-C fallbacks={fallback_measurements}; "
        f"JSON={args.output}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
