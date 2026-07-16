#!/usr/bin/env python3
"""Verify Megalodon JAX modeling semantics against local authoritative sources.

This standalone verifier compares the JAX implementation against independent mathematical
oracles and, when available, the exact local original PyTorch/CUDA source. Repository-only
checks remain runnable without untracked evidence files; source-anchoring checks are reported
as skipped. The verifier does not build the fused CUDA extension, import downstream ports, or
import an installed Megalodon package.

Usage:
    python tools/verify_modeling_correctness.py \
        --jax-repo /path/to/megalodon-jax

By default the process exits non-zero when an invariant fails. Use --no-fail to
produce a report without failing CI. Use --json PATH to save machine-readable
results. Use --include-slow to add model forward/cache checks that trigger substantial JAX
compilation on CPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# These must be set before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


def _max_abs_error(actual: Any, expected: Any) -> float:
    """Return the maximum absolute difference between array-like values."""
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


@dataclass
class CheckResult:
    name: str
    status: Literal["passed", "failed", "skipped"]
    passed: bool | None
    severity: str
    summary: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class Audit:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def record(
        self,
        name: str,
        passed: bool,
        severity: str,
        summary: str,
        **details: Any,
    ) -> None:
        self.results.append(
            CheckResult(
                name=name,
                status="passed" if passed else "failed",
                passed=bool(passed),
                severity=severity,
                summary=summary,
                details=_jsonable(details),
            )
        )

    def skip(self, name: str, severity: str, summary: str, **details: Any) -> None:
        """Record a check that was intentionally not executed."""
        self.results.append(
            CheckResult(
                name=name,
                status="skipped",
                passed=None,
                severity=severity,
                summary=summary,
                details=_jsonable(details),
            )
        )

    def run(
        self, name: str, severity: str, fn: Callable[[], tuple[bool, str, dict[str, Any]]]
    ) -> None:
        try:
            passed, summary, details = fn()
            self.record(name, passed, severity, summary, **details)
        except Exception as exc:  # diagnostic should report all checks, not stop at first
            self.record(
                name,
                False,
                severity,
                f"check raised {type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )


def _jsonable(value: Any) -> Any:
    if isinstance(value, type):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    try:
        # JAX arrays support __array__ when concrete.
        arr = np.asarray(value)
        if arr.ndim == 0:
            item = arr.item()
            if isinstance(item, (str, int, float, bool)) or item is None:
                return item
            return str(item)
        return arr.tolist()
    except Exception:
        return str(value)


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a local evidence file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(repo: Path) -> str:
    """Return the repository commit without importing a Git library."""
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


def _torch_environment() -> dict[str, Any]:
    """Return PyTorch/CUDA environment facts without making Torch mandatory."""
    try:
        import torch
    except ImportError:
        return {"version": None, "cuda_available": False, "devices": []}
    devices = []
    if torch.cuda.is_available():
        devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    return {
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "devices": devices,
    }


def exact_timestep_norm(
    x: np.ndarray,
    groups: int,
    *,
    mask: np.ndarray | None = None,
    segment_ids: np.ndarray | None = None,
    eps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Literal causal group-normalization oracle from paper equation (4).

    Count is in timesteps, while each group moment includes all group-feature
    scalars from every valid timestep. Segment IDs reset the state at a boundary;
    ID 0 is padding. Invalid positions emit exact zeros, matching upstream CUDA.
    """

    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected (B,L,D), got {x.shape}")
    batch, length, dim = x.shape
    if dim % groups:
        raise ValueError("dim must be divisible by groups")
    group_size = dim // groups
    if mask is None:
        mask = np.ones((batch, length), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    if segment_ids is not None:
        segment_ids = np.asarray(segment_ids)

    out = np.zeros_like(x, dtype=np.float64)
    final_count = np.zeros(batch, dtype=np.int32)
    final_mean = np.zeros((batch, groups), dtype=np.float64)
    final_var = np.ones((batch, groups), dtype=np.float64)

    for b in range(batch):
        values: list[list[float]] = [[] for _ in range(groups)]
        count = 0
        previous_segment: int | None = None
        last_valid_state = (0, np.zeros(groups), np.ones(groups))

        for t in range(length):
            seg = None if segment_ids is None else int(segment_ids[b, t])
            valid = bool(mask[b, t]) and (segment_ids is None or seg > 0)

            if segment_ids is not None:
                boundary = t == 0 or seg != previous_segment
                if boundary:
                    values = [[] for _ in range(groups)]
                    count = 0
                previous_segment = seg

            if valid:
                count += 1
                for g in range(groups):
                    start = g * group_size
                    stop = start + group_size
                    values[g].extend(x[b, t, start:stop].tolist())

            means = np.zeros(groups, dtype=np.float64)
            variances = np.ones(groups, dtype=np.float64)
            for g in range(groups):
                if values[g]:
                    means[g] = float(np.mean(values[g]))
                    variances[g] = float(np.var(values[g], ddof=0))

            if valid:
                for g in range(groups):
                    start = g * group_size
                    stop = start + group_size
                    out[b, t, start:stop] = (x[b, t, start:stop] - means[g]) / math.sqrt(
                        variances[g] + eps
                    )
                last_valid_state = (count, means.copy(), variances.copy())
            else:
                out[b, t, :] = 0.0

        final_count[b] = last_valid_state[0]
        final_mean[b] = last_valid_state[1]
        final_var[b] = last_valid_state[2]

    return out, final_count, final_mean, final_var


def adjacent_pair_rope(x: np.ndarray, positions: np.ndarray, base: float) -> np.ndarray:
    """NumPy oracle for upstream view_as_complex(...reshape(...,-1,2)) RoPE."""

    x = np.asarray(x, dtype=np.float32)
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("RoPE dim must be even")
    half = dim // 2
    inv = np.exp(np.arange(half, dtype=np.float32) * (-np.log(base) / half))
    positions = np.asarray(positions, dtype=np.float32)
    angles = positions[..., None] * inv
    cos = np.cos(angles)
    sin = np.sin(angles)

    # x is B,L,H,D. Broadcast B,L,1,D/2.
    while cos.ndim < x.ndim - 1:
        cos = np.expand_dims(cos, axis=-2)
        sin = np.expand_dims(sin, axis=-2)
    pairs = x.reshape(*x.shape[:-1], half, 2)
    real = pairs[..., 0]
    imag = pairs[..., 1]
    out = np.stack([real * cos - imag * sin, imag * cos + real * sin], axis=-1)
    return out.reshape(x.shape).astype(x.dtype)


def source_parameter_count(
    *,
    model_dim: int,
    num_layers: int,
    num_heads: int,
    z_dim: int,
    value_dim: int,
    ffn_hidden_dim: int,
    cema_ndim: int,
    vocab_size: int,
    swiglu: bool,
    share_emb: bool,
) -> int:
    """Exact world-size-1 trainable count for uploaded upstream architecture."""

    del num_heads  # does not affect total Q/K widths
    d, z, v, f, n = model_dim, z_dim, value_dim, ffn_hidden_dim, cema_ndim
    timestep_norm = 2 * d
    cema = 4 * d * n + 2 * d
    rmsnorm = d
    attention_projections = (
        (d * z + z) + (d * v + v) + (d * v + v) + (d * d + d) + (v * d)  # wh2 has no bias
    )
    qk_affine = 4 * z
    ffn_norm = 2 * d
    ffn = d * f + f * d + (d * f if swiglu else 0)  # no FFN biases
    per_layer = timestep_norm + cema + rmsnorm + attention_projections + qk_affine + ffn_norm + ffn
    embedding = vocab_size * d
    output = 0 if share_emb else vocab_size * d
    final_norm = 2 * d
    return embedding + num_layers * per_layer + final_norm + output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jax-repo",
        type=Path,
        default=Path.cwd(),
        help="Root of the megalodon-jax repository (must contain src/megalodon_jax).",
    )
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=None,
        help=(
            "Exact local original PyTorch/CUDA repository; source checks use the "
            "local-scratch default when present and skip otherwise"
        ),
    )
    parser.add_argument(
        "--backend",
        choices=("cpu", "gpu"),
        default="cpu",
        help="JAX backend for executable checks.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--include-slow", action="store_true", help="Include JIT-heavy forward/cache checks."
    )
    parser.add_argument("--no-fail", action="store_true", help="Always exit zero after reporting.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["JAX_PLATFORM_NAME"] = args.backend
    repo = args.jax_repo.resolve()
    upstream = (
        args.upstream_repo.resolve()
        if args.upstream_repo is not None
        else repo / "local-scratch" / "megalodon-upstream-cuda_torch"
    )
    src = repo / "src"
    package = src / "megalodon_jax"
    if not package.is_dir():
        print(f"ERROR: {package} does not exist", file=sys.stderr)
        return 2
    if args.upstream_repo is not None and not upstream.is_dir():
        print(f"ERROR: local upstream repository does not exist: {upstream}", file=sys.stderr)
        return 2
    upstream_available = upstream.is_dir()
    upstream_config_path = upstream / "megalodon" / "config.py"
    repository_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository_root))
    sys.path.insert(0, str(src))

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from megalodon_jax.checkpoint import model_state_dict
    from megalodon_jax.config import MegalodonConfig
    from megalodon_jax.layers import RotaryEmbedding, TimestepNorm
    from megalodon_jax.layers.norms import BatchedLayerNorm
    from megalodon_jax.model import MegalodonForCausalLM
    from megalodon_jax.utils import get_initializer
    from tests.reference.cache import cache_partition_errors
    from tests.reference.training import deterministic_tiny_overfit
    from tests.reference.upstream import (
        differentiable_state,
        source_forward,
        trainable_upstream_keys,
    )

    def tiny_config(**overrides: Any) -> MegalodonConfig:
        """Return the canonical verifier config with explicit overrides."""
        config = MegalodonConfig(
            vocab_size=32,
            model_dim=16,
            num_layers=1,
            num_heads=1,
            z_dim=8,
            value_dim=16,
            ffn_hidden_dim=32,
            cema_ndim=2,
            chunk_size=4,
            norm_num_groups=4,
        )
        return dataclasses.replace(config, **overrides)

    audit = Audit()

    def check_timestep_math() -> tuple[bool, str, dict[str, Any]]:
        x = np.array([[[1.0, 3.0, 2.0, 6.0], [5.0, 7.0, 10.0, 14.0]]], dtype=np.float32)
        module = TimestepNorm(4, 2, eps=1e-8)
        actual, state = module(jnp.asarray(x))
        expected, count, mean, var = exact_timestep_norm(x, 2, eps=1e-8)
        y_err = _max_abs_error(actual, expected)
        var_err = _max_abs_error(state.var, var)
        passed = (
            y_err <= 2e-6 and var_err <= 2e-6 and np.array_equal(np.asarray(state.count), count)
        )
        return (
            passed,
            "TimestepNorm matches scalar paper/CUDA moments"
            if passed
            else "TimestepNorm omits within-group feature variance or has wrong empty-state M2",
            {
                "max_abs_output_error": y_err,
                "max_abs_state_var_error": var_err,
                "actual_output": np.asarray(actual),
                "expected_output": expected,
                "actual_state_var": np.asarray(state.var),
                "expected_state_var": var,
                "actual_state_mean": np.asarray(state.mean),
                "expected_state_mean": mean,
            },
        )

    audit.run("timestep_norm_scalar_moments", "P0", check_timestep_math)

    def check_timestep_mask() -> tuple[bool, str, dict[str, Any]]:
        x = np.array([[[1.0, 3.0, 2.0, 6.0], [100.0, 200.0, 300.0, 400.0]]], dtype=np.float32)
        mask = np.array([[True, False]])
        module = TimestepNorm(4, 2, eps=1e-8)
        actual, state = module(jnp.asarray(x), mask=jnp.asarray(mask))
        expected, count, mean, var = exact_timestep_norm(x, 2, mask=mask, eps=1e-8)
        masked = np.asarray(actual)[0, 1]
        passed = np.array_equal(masked, np.zeros_like(masked)) and np.array_equal(
            np.asarray(state.count), count
        )
        return (
            passed,
            "Masked outputs are exact zero and state is unchanged"
            if passed
            else "Masked TimestepNorm positions leak nonzero activations",
            {
                "actual_masked_output": masked,
                "expected_masked_output": expected[0, 1],
                "actual_count": np.asarray(state.count),
                "expected_count": count,
                "expected_mean": mean,
                "expected_var": var,
            },
        )

    audit.run("timestep_norm_mask_zero", "P0", check_timestep_mask)

    def check_timestep_segments() -> tuple[bool, str, dict[str, Any]]:
        x = np.array(
            [
                [
                    [1.0, 3.0, 2.0, 6.0],
                    [5.0, 7.0, 10.0, 14.0],
                    [2.0, 4.0, 8.0, 12.0],
                    [6.0, 8.0, 16.0, 20.0],
                ]
            ],
            dtype=np.float32,
        )
        segments = np.array([[1, 1, 2, 2]], dtype=np.int32)
        module = TimestepNorm(4, 2, eps=1e-8)
        actual, state = module(jnp.asarray(x), segment_ids=jnp.asarray(segments))
        expected, count, mean, var = exact_timestep_norm(
            x,
            2,
            segment_ids=segments,
            eps=1e-8,
        )
        err = _max_abs_error(actual, expected)
        passed = (
            err <= 2e-6
            and np.array_equal(np.asarray(state.count), count)
            and np.allclose(np.asarray(state.var), var, atol=2e-6)
        )
        return (
            passed,
            "Segment resets match independent-document scalar oracle"
            if passed
            else "Segmented TimestepNorm inherits incorrect variance/reset baseline",
            {
                "max_abs_output_error": err,
                "actual_final_count": np.asarray(state.count),
                "expected_final_count": count,
                "actual_final_var": np.asarray(state.var),
                "expected_final_var": var,
                "expected_final_mean": mean,
            },
        )

    audit.run("timestep_norm_segment_reset", "P0", check_timestep_segments)

    def check_rope_layout() -> tuple[bool, str, dict[str, Any]]:
        q = np.arange(1, 9, dtype=np.float32).reshape(1, 1, 1, 8)
        k = np.arange(11, 19, dtype=np.float32).reshape(1, 1, 1, 8)
        module = RotaryEmbedding(8, base=10000.0)
        q_actual, k_actual = module(
            q=jnp.asarray(q), k=jnp.asarray(k), start_index=jnp.asarray(7, jnp.int32)
        )
        pos = np.array([[7]], dtype=np.float32)
        q_expected = adjacent_pair_rope(q, pos, 10000.0)
        k_expected = adjacent_pair_rope(k, pos, 10000.0)
        q_err = _max_abs_error(q_actual, q_expected)
        k_err = _max_abs_error(k_actual, k_expected)
        passed = q_err <= 2e-6 and k_err <= 2e-6
        return (
            passed,
            "RoPE uses upstream adjacent coordinate pairs"
            if passed
            else "RoPE uses a different half-split coordinate basis",
            {
                "max_abs_q_error": q_err,
                "max_abs_k_error": k_err,
                "actual_q": np.asarray(q_actual).reshape(-1),
                "expected_q": q_expected.reshape(-1),
            },
        )

    audit.run("rope_adjacent_pair_layout", "P0", check_rope_layout)

    def check_rope_trainability() -> tuple[bool, str, dict[str, Any]]:
        module = RotaryEmbedding(8, base=10000.0)
        array_leaves = [leaf for leaf in jax.tree.leaves(module) if eqx.is_array(leaf)]
        passed = not array_leaves
        return (
            passed,
            "RoPE contains no trainable array leaves; frequencies are derived"
            if passed
            else "RoPE contains array leaves that will enter generic Equinox optimizers",
            {
                "array_leaf_count": len(array_leaves),
                "array_leaves": [
                    {"shape": list(leaf.shape), "dtype": str(leaf.dtype)} for leaf in array_leaves
                ],
            },
        )

    audit.run("rope_frequency_is_buffer", "P0", check_rope_trainability)

    def check_layernorm_storage() -> tuple[bool, str, dict[str, Any]]:
        module = BatchedLayerNorm(4, eps=1e-5, affine=True)
        stored = np.asarray(module.weight)
        x = jnp.asarray([[[1.0, 2.0, 4.0, 8.0]]], dtype=jnp.float32)
        y = np.asarray(module(x))
        x_np = np.asarray(x, dtype=np.float32)
        mean = x_np.mean(axis=-1, keepdims=True)
        var = ((x_np - mean) ** 2).mean(axis=-1, keepdims=True)
        identity_expected = (x_np - mean) / np.sqrt(var + 1e-5)
        effective_identity = _max_abs_error(y, identity_expected) <= 2e-6
        # Source-compatible storage must be zero while effective scale is one.
        passed = np.array_equal(stored, np.zeros_like(stored)) and effective_identity
        return (
            passed,
            "LayerNorm stores zero and applies weight+1"
            if passed
            else "LayerNorm uses direct one-scale storage; upstream checkpoint zeros would collapse output",
            {
                "stored_weight": stored,
                "effective_identity_error": _max_abs_error(y, identity_expected),
            },
        )

    audit.run("layernorm_plus_one_storage", "P0", check_layernorm_storage)

    def check_explicit_tying() -> tuple[bool, str, dict[str, Any]]:
        fields = {field.name for field in dataclasses.fields(MegalodonConfig)}
        has_flag = "share_emb" in fields
        cfg = tiny_config(output_size=32)
        model = MegalodonForCausalLM(cfg, key=jax.random.PRNGKey(1))
        # Source default is untied even when output width equals vocab width.
        passed = has_flag and not bool(model.tied)
        return (
            passed,
            "Embedding sharing is an explicit independent flag"
            if passed
            else "Vocabulary-sized output is forcibly tied and upstream untied default is unrepresentable",
            {
                "has_share_emb_field": has_flag,
                "model_tied": bool(model.tied),
                "lm_head_present": model.lm_head is not None,
            },
        )

    audit.run("explicit_output_weight_tying", "P0", check_explicit_tying)

    def check_7b_preset() -> tuple[bool, str, dict[str, Any]]:
        configs = {
            "mega200M": MegalodonConfig.from_upstream_mega200m(vocab_size=32_000),
            "mega1.3B": MegalodonConfig.from_upstream_mega1_3b(vocab_size=32_000),
            "mega1.3B_pg19": MegalodonConfig.from_upstream_mega1_3b_pg19(vocab_size=32_000),
            "mega7.1B": MegalodonConfig.from_upstream_mega7_1b(vocab_size=32_000),
            "mega7.3B": MegalodonConfig.from_upstream_mega7_3b(vocab_size=32_000),
            "paper7B": MegalodonConfig.from_paper_7b(),
        }
        expected_counts = {
            "mega200M": 220_627_968,
            "mega1.3B": 1_342_832_640,
            "mega1.3B_pg19": 1_327_628_288,
            "mega7.1B": 7_117_381_632,
            "mega7.3B": 7_385_817_088,
            "paper7B": 7_385_817_088,
        }
        counts = {
            name: config.parameter_count_breakdown()["total"] for name, config in configs.items()
        }
        source_counts = {
            name: source_parameter_count(
                model_dim=config.model_dim,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                z_dim=config.z_dim,
                value_dim=config.value_dim,
                ffn_hidden_dim=config.ffn_hidden_dim,
                cema_ndim=config.cema_ndim,
                vocab_size=config.vocab_size,
                swiglu=config.swiglu,
                share_emb=config.share_emb,
            )
            for name, config in configs.items()
        }
        try:
            MegalodonConfig.from_7b()
        except ValueError:
            ambiguous_rejected = True
        else:
            ambiguous_rejected = False
        identities = {
            "mega7.1B_non_swiglu_11264": not configs["mega7.1B"].swiglu
            and configs["mega7.1B"].ffn_hidden_dim == 11_264,
            "mega7.3B_swiglu_8192": configs["mega7.3B"].swiglu
            and configs["mega7.3B"].ffn_hidden_dim == 8_192,
            "paper_chunk_and_base": configs["paper7B"].chunk_size == 4_096
            and configs["paper7B"].effective_rope_base == 100_000.0,
        }
        passed = (
            counts == expected_counts
            and source_counts == counts
            and ambiguous_rejected
            and all(identities.values())
        )
        return (
            passed,
            "Named source/paper presets and exact parameter counts are unambiguous",
            {
                "actual_counts": counts,
                "independent_source_formula_counts": source_counts,
                "expected_counts": expected_counts,
                "ambiguous_from_7b_rejected": ambiguous_rejected,
                "identities": identities,
            },
        )

    audit.run("source_compatible_7b_preset", "P0", check_7b_preset)

    def check_default_init_mode() -> tuple[bool, str, dict[str, Any]]:
        cfg = MegalodonConfig()
        passed = cfg.init_mode == "he"
        return (
            passed,
            "Default internal initialization is upstream he"
            if passed
            else "Default internal initialization is Gaussian; current model reinit uses unit-scale weights",
            {"actual_default_init_mode": cfg.init_mode, "expected": "he"},
        )

    audit.run("default_initialization_mode", "P0", check_default_init_mode)

    def check_he_distribution() -> tuple[bool, str, dict[str, Any]]:
        # Equinox Linear stores weights as (out_features, in_features), while
        # jax.nn.initializers.he_normal() defaults to in_axis=-2/out_axis=-1,
        # the opposite convention. The source initializer also uses gain for
        # kaiming_normal_(a=sqrt(5)), not canonical ReLU He gain.
        shapes = {
            "wz": (256, 1024),
            "wv": (2048, 1024),
            "wr": (2048, 1024),
            "wh1": (1024, 1024),
            "wh2": (1024, 2048),
            "fc1": (2560, 1024),
            "fc2": (1024, 2560),
            "fc3": (2560, 1024),
        }
        initializer = get_initializer("he")
        rows: dict[str, Any] = {}
        passed = True
        for index, (name, shape) in enumerate(shapes.items()):
            weights = np.asarray(initializer(jax.random.PRNGKey(42 + index), shape, jnp.float32))
            actual_std = float(weights.std(ddof=0))
            true_fan_in = shape[-1]
            source_target = 1.0 / math.sqrt(3.0 * true_fan_in)
            ratio = actual_std / source_target
            theoretical_current = math.sqrt(2.0 / shape[-2])
            rows[name] = {
                "shape_out_in": list(shape),
                "actual_std": actual_std,
                "source_target_std": source_target,
                "actual_to_source_ratio": ratio,
                "jax_default_interpreted_fan_in": shape[-2],
                "true_linear_fan_in": true_fan_in,
                "theoretical_current_std": theoretical_current,
            }
            passed = passed and abs(ratio - 1.0) <= 0.06
        return (
            passed,
            "He initializer matches source gain and Equinox fan-in axis"
            if passed
            else "JAX he_normal has both the wrong gain and the wrong fan-in axis for rectangular Equinox weights",
            {"layers": rows},
        )

    audit.run("he_initialization_gain_and_axis", "P0", check_he_distribution)

    def check_other_initializer_distributions() -> tuple[bool, str, dict[str, Any]]:
        shape = (2048, 1024)
        fan_in, fan_out = shape[-1], shape[-2]
        truncated_variance = 0.9733369246625415
        cases = {
            "xavier": (
                get_initializer("xavier")(jax.random.PRNGKey(81), shape, jnp.float32),
                math.sqrt(2.0 / (fan_in + fan_out)),
            ),
            "bert": (
                get_initializer("bert")(jax.random.PRNGKey(82), shape, jnp.float32),
                0.02,
            ),
            "gaussian": (
                get_initializer("gaussian")(jax.random.PRNGKey(83), shape, jnp.float32),
                math.sqrt(truncated_variance),
            ),
        }
        rows = {}
        passed = True
        for name, (values, expected_std) in cases.items():
            array = np.asarray(values)
            actual_std = float(array.std(ddof=0))
            ratio = actual_std / expected_std
            rows[name] = {
                "shape_out_in": shape,
                "actual_std": actual_std,
                "expected_std": expected_std,
                "actual_to_expected_ratio": ratio,
                "minimum": float(array.min()),
                "maximum": float(array.max()),
            }
            passed = passed and abs(ratio - 1.0) <= 0.03
        xavier_bound = math.sqrt(6.0 / (fan_in + fan_out))
        passed = passed and rows["xavier"]["minimum"] >= -xavier_bound
        passed = passed and rows["xavier"]["maximum"] <= xavier_bound
        passed = passed and rows["gaussian"]["minimum"] >= -3.0
        passed = passed and rows["gaussian"]["maximum"] <= 3.0
        return (
            passed,
            "Xavier, BERT, and Gaussian distributions match released contracts",
            {"distributions": rows, "xavier_bound": xavier_bound},
        )

    audit.run(
        "other_initialization_distributions",
        "P0",
        check_other_initializer_distributions,
    )

    def check_embedding_init_policy() -> tuple[bool, str, dict[str, Any]]:
        cfg = MegalodonConfig(
            vocab_size=4096,
            model_dim=128,
            num_layers=1,
            num_heads=1,
            z_dim=32,
            value_dim=256,
            ffn_hidden_dim=384,
            cema_ndim=2,
            chunk_size=4,
            norm_num_groups=8,
            init_mode="he",
        )
        model = MegalodonForCausalLM(cfg, key=jax.random.PRNGKey(101))
        actual_std = float(np.asarray(model.model.embed.weight).std(ddof=0))
        source_target = 1.0 / math.sqrt(cfg.model_dim)
        ratio = actual_std / source_target
        passed = abs(ratio - 1.0) <= 0.08
        return (
            passed,
            "Embedding uses fixed source Gaussian independent of internal init mode"
            if passed
            else "Global init_mode reaches the embedding; downstream he configs therefore do not reproduce source embedding initialization",
            {
                "shape": list(model.model.embed.weight.shape),
                "actual_std": actual_std,
                "source_target_std": source_target,
                "ratio": ratio,
                "current_he_theoretical_std": math.sqrt(2.0 / cfg.vocab_size),
                "model_tied": bool(model.tied),
            },
        )

    audit.run("embedding_initialization_policy", "P0", check_embedding_init_policy)

    def check_output_init_policy() -> tuple[bool, str, dict[str, Any]]:
        base = dict(
            vocab_size=4096,
            output_size=2048,
            model_dim=128,
            num_layers=1,
            num_heads=1,
            z_dim=32,
            value_dim=256,
            ffn_hidden_dim=384,
            cema_ndim=2,
            chunk_size=4,
            norm_num_groups=8,
        )
        model_he = MegalodonForCausalLM(
            MegalodonConfig(**base, init_mode="he"), key=jax.random.PRNGKey(102)
        )
        model_gaussian = MegalodonForCausalLM(
            MegalodonConfig(**base, init_mode="gaussian"), key=jax.random.PRNGKey(103)
        )
        source_target = 1.0 / math.sqrt(base["model_dim"])
        actual_he = float(np.asarray(model_he.lm_head.weight).std(ddof=0))
        actual_gaussian = float(np.asarray(model_gaussian.lm_head.weight).std(ddof=0))
        he_ratio = actual_he / source_target
        gaussian_ratio = actual_gaussian / source_target
        passed = abs(he_ratio - 1.0) <= 0.08 and abs(gaussian_ratio - 1.0) <= 0.08
        return (
            passed,
            "Untied head uses fixed source Gaussian based on model_dim"
            if passed
            else "Untied head is coupled to init_mode and Gaussian uses output_size instead of model_dim",
            {
                "shape": list(model_he.lm_head.weight.shape),
                "source_target_std": source_target,
                "he_actual_std": actual_he,
                "he_ratio": he_ratio,
                "gaussian_actual_std": actual_gaussian,
                "gaussian_ratio": gaussian_ratio,
            },
        )

    audit.run("output_head_initialization_policy", "P0", check_output_init_policy)

    def _read_required(path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8")

    def check_upstream_init_contract() -> tuple[bool, str, dict[str, Any]]:
        cfg_text = _read_required(upstream / "megalodon" / "config.py")
        util_text = _read_required(upstream / "megalodon" / "utils.py")
        model_text = _read_required(upstream / "megalodon" / "model" / "mega.py")
        facts = {
            "internal_default_he": "init_mode: str = 'he'" in cfg_text,
            "source_kaiming_a_sqrt5": "a = math.sqrt(5.0)" in util_text
            and "kaiming_normal_" in util_text,
            "embedding_fixed_gaussian_model_dim": "get_init_fn('gaussian', dim=self.model_dim)"
            in model_text,
            "output_fixed_gaussian_model_dim": "init_fn = get_init_fn('gaussian', dim=self.model_dim)"
            in model_text,
        }
        passed = all(facts.values())
        summary = (
            "Local source splits internal He from embedding/head Gaussian"
            if passed
            else "Could not establish the complete local-source initialization contract"
        )
        return passed, summary, facts

    if upstream_available:
        audit.run("upstream_initialization_contract", "INFO", check_upstream_init_contract)
    else:
        audit.skip(
            "upstream_initialization_contract",
            "INFO",
            "Local upstream source is unavailable; repository-only checks still ran",
            expected_path=upstream,
        )

    def check_torch_transcription_source_contract() -> tuple[bool, str, dict[str, Any]]:
        attention_path = upstream / "megalodon" / "modules" / "moving_average_gated_attention.py"
        cema_path = upstream / "megalodon" / "modules" / "complex_exponential_moving_average.py"
        ffn_path = upstream / "megalodon" / "modules" / "normalized_feedforward_network.py"
        rope_path = upstream / "megalodon" / "modules" / "rotary_positional_embedding.py"
        attention = _read_required(attention_path)
        cema = _read_required(cema_path)
        ffn = _read_required(ffn_path)
        rope = _read_required(rope_path)
        facts = {
            "unscaled_attention_scores": "scores = torch.matmul(xq, xk.transpose(2, 3))"
            in attention,
            "fp32_attention_softmax": "F.softmax(scores, dim=-1, dtype=torch.float32)" in attention,
            "attention_two_projection_sum": "self.wh1(mx) + self.wh2(attn)" in attention,
            "attention_residual": "out = h + residual" in attention,
            "cema_complex_polar": "q = torch.polar(1.0 - alpha * delta, theta)" in cema,
            "swiglu_order": "F.silu(self.fc1(x)) * self.fc3(x)" in ffn,
            "two_hop_ffn_residual": "self.rescale(x) + residual" in ffn,
            "adjacent_pair_rope": "torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))"
            in rope,
        }
        passed = all(facts.values())
        return (
            passed,
            "Pure-Torch transcription is anchored to the exact released source equations",
            {
                **facts,
                "source_sha256": {
                    "attention": _sha256(attention_path),
                    "cema": _sha256(cema_path),
                    "ffn": _sha256(ffn_path),
                    "rope": _sha256(rope_path),
                },
            },
        )

    if upstream_available:
        audit.run(
            "torch_transcription_source_contract",
            "INFO",
            check_torch_transcription_source_contract,
        )
    else:
        audit.skip(
            "torch_transcription_source_contract",
            "INFO",
            "Local upstream source is unavailable; the bundled transcription was not source-anchored",
            expected_path=upstream,
        )

    def check_cuda_source_mask_contract() -> tuple[bool, str, dict[str, Any]]:
        source_path = upstream / "megalodon" / "csrc" / "ops" / "timestep_norm_kernel.cu"
        source = _read_required(source_path)
        facts = {
            "has_padding_mask_branch": "padding_mask" in source,
            "writes_zero_for_padding": "y_ptr[i] = T(0)" in source,
        }
        passed = all(facts.values())
        return (
            passed,
            "CUDA source checked; building and runtime execution are out of scope",
            {
                "verification_mode": "source_only",
                "source": str(source_path),
                "sha256": _sha256(source_path),
                **facts,
            },
        )

    if upstream_available:
        audit.run("cuda_timestep_mask_source_contract", "INFO", check_cuda_source_mask_contract)
    else:
        audit.skip(
            "cuda_timestep_mask_source_contract",
            "INFO",
            "Local upstream CUDA source is unavailable",
            expected_path=upstream,
        )

    def check_source_biases() -> tuple[bool, str, dict[str, Any]]:
        cfg = tiny_config(swiglu=True)
        model = MegalodonForCausalLM(cfg, key=jax.random.PRNGKey(3))
        layer = model.model.layers[0]
        present = {
            "attn.wh2": layer.attn.wh2.bias is not None,
            "ffn.fc1": layer.ffn.fc1.bias is not None,
            "ffn.fc2": layer.ffn.fc2.bias is not None,
            "ffn.fc3": layer.ffn.fc3 is not None and layer.ffn.fc3.bias is not None,
        }
        passed = not any(present.values())
        return (
            passed,
            "Source no-bias projections are preserved"
            if passed
            else "JAX adds biases to FFN and/or attention wh2",
            {"unexpected_bias_present": present},
        )

    audit.run("source_projection_bias_flags", "P1", check_source_biases)

    def check_dropout_bounds() -> tuple[bool, str, dict[str, Any]]:
        accepted: list[str] = []
        for name in ("dropout", "attention_dropout", "hidden_dropout"):
            try:
                MegalodonConfig(**{name: 1.0})
                accepted.append(name)
            except (ValueError, AssertionError):
                pass
        passed = not accepted
        return (
            passed,
            "All dropout probabilities require p < 1"
            if passed
            else "p=1.0 is accepted and will divide by zero",
            {"p_equals_one_accepted_for": accepted},
        )

    audit.run("dropout_upper_bound", "P1", check_dropout_bounds)

    def check_fp16_rejection() -> tuple[bool, str, dict[str, Any]]:
        rejected: dict[str, bool] = {}
        for field in (
            "param_dtype",
            "compute_dtype",
            "accum_dtype",
            "attention_softmax_dtype",
            "loss_softmax_dtype",
        ):
            try:
                MegalodonConfig(**{field: jnp.float16})
            except ValueError:
                rejected[field] = True
            else:
                rejected[field] = False
        try:
            TimestepNorm(4, 2)(jnp.ones((1, 1, 4), dtype=jnp.float16))
        except TypeError:
            rejected["timestep_norm_input"] = True
        else:
            rejected["timestep_norm_input"] = False
        passed = all(rejected.values())
        return (
            passed,
            "FP16 is rejected; the supported numerical surface is FP32/BF16 only"
            if passed
            else "An FP16 configuration or state path remains reachable",
            {"rejected": rejected, "supported": ["float32", "bfloat16"]},
        )

    audit.run("fp16_is_out_of_scope", "P1", check_fp16_rejection)

    if args.include_slow:

        def tiny_reference_fixture() -> tuple[Any, Any, Any, np.ndarray]:
            config = MegalodonConfig(
                vocab_size=32,
                model_dim=8,
                num_layers=1,
                num_heads=2,
                z_dim=8,
                value_dim=8,
                ffn_hidden_dim=16,
                cema_ndim=2,
                chunk_size=4,
                norm_num_groups=2,
                swiglu=True,
                rescale_nffn=True,
                scale_emb=True,
                share_emb=False,
            )
            model = MegalodonForCausalLM(config, key=jax.random.PRNGKey(8675309))
            token_values = np.asarray([[1, 7, 3, 11], [5, 2, 13, 17]], dtype=np.int32)
            loss_weights = np.linspace(
                -0.75,
                0.9,
                token_values.shape[0] * token_values.shape[1] * config.effective_output_size,
                dtype=np.float32,
            ).reshape(token_values.shape[0], token_values.shape[1], config.effective_output_size)
            return config, model, token_values, loss_weights

        def make_torch_reference_state(model: Any) -> tuple[dict[str, Any], list[str]]:
            from megalodon_jax.convert import export_upstream_state_dict

            source_state = export_upstream_state_dict(model)
            trainable = list(trainable_upstream_keys(source_state))
            state = differentiable_state(source_state)
            return state, trainable

        def check_logits_dtype() -> tuple[bool, str, dict[str, Any]]:
            cfg = tiny_config(
                param_dtype=jnp.float32,
                compute_dtype=jnp.bfloat16,
                accum_dtype=jnp.float32,
            )
            model = MegalodonForCausalLM(cfg, key=jax.random.PRNGKey(8))
            logits, _ = model(jnp.asarray([[1, 2]], dtype=jnp.int32))
            passed = logits.dtype == jnp.float32
            return (
                passed,
                "Returned logits preserve upstream fp32 contract"
                if passed
                else "Returned logits are downcast to compute dtype",
                {"actual_logits_dtype": str(logits.dtype), "expected": "float32"},
            )

        audit.run("fp32_output_logits", "P1", check_logits_dtype)

        def check_torch_forward_and_gradients() -> tuple[bool, str, dict[str, Any]]:
            import torch

            from megalodon_jax.convert import export_upstream_state_dict

            config, model, token_values, loss_weights = tiny_reference_fixture()
            jax_tokens = jnp.asarray(token_values)
            jax_weights = jnp.asarray(loss_weights)

            def jax_objective(candidate: Any) -> tuple[Any, Any]:
                logits, _ = candidate(jax_tokens)
                return jnp.sum(logits * jax_weights), logits

            (jax_loss, jax_logits), jax_grads = eqx.filter_value_and_grad(
                jax_objective,
                has_aux=True,
            )(model)

            torch_state, trainable = make_torch_reference_state(model)
            torch_tokens = torch.from_numpy(token_values)
            torch_weights = torch.from_numpy(loss_weights)
            torch_logits = source_forward(torch_tokens, torch_state, config)
            torch_loss = torch.sum(torch_logits * torch_weights)
            torch_loss.backward()

            gradient_state = export_upstream_state_dict(jax_grads)
            gradient_errors: dict[str, Any] = {}
            gradients_pass = True
            for name in trainable:
                reference_gradient = torch_state[name].grad
                if reference_gradient is None:
                    gradient_errors[name] = {"missing_torch_gradient": True}
                    gradients_pass = False
                    continue
                actual = gradient_state[name].detach().cpu().numpy()
                expected = reference_gradient.detach().cpu().numpy()
                absolute = np.abs(actual - expected)
                max_abs = float(absolute.max(initial=0.0))
                scale = np.maximum(np.maximum(np.abs(actual), np.abs(expected)), 1e-6)
                max_rel = float((absolute / scale).max(initial=0.0))
                close = bool(np.allclose(actual, expected, rtol=5e-3, atol=2e-4))
                gradient_errors[name] = {
                    "max_abs_error": max_abs,
                    "max_relative_error": max_rel,
                    "passed": close,
                }
                gradients_pass = gradients_pass and close

            jax_logits_np = np.asarray(jax_logits)
            torch_logits_np = torch_logits.detach().cpu().numpy()
            output_error = _max_abs_error(jax_logits_np, torch_logits_np)
            loss_error = abs(float(jax_loss) - float(torch_loss.detach()))
            forward_pass = bool(np.allclose(jax_logits_np, torch_logits_np, rtol=5e-4, atol=5e-5))
            passed = forward_pass and gradients_pass
            return (
                passed,
                "Tiny JAX logits and every trainable gradient match the source-derived Torch transcription"
                if passed
                else "Tiny source-derived Torch/JAX forward or gradient parity failed",
                {
                    "max_abs_logits_error": output_error,
                    "absolute_loss_error": loss_error,
                    "gradient_tolerances": {"rtol": 5e-3, "atol": 2e-4},
                    "logit_tolerances": {"rtol": 5e-4, "atol": 5e-5},
                    "gradient_errors": gradient_errors,
                },
            )

        torch_available = importlib.util.find_spec("torch") is not None
        # The bundled Torch path is a same-author transcription: useful consistency evidence,
        # but not an independent ground-truth implementation.
        if torch_available:
            audit.run(
                "source_torch_jax_forward_gradient_parity",
                "INFO",
                check_torch_forward_and_gradients,
            )
        else:
            audit.skip(
                "source_torch_jax_forward_gradient_parity",
                "INFO",
                "PyTorch is unavailable; install the convert extra to run parity",
                required_extra="convert",
            )

        def check_short_adamw_parity() -> tuple[bool, str, dict[str, Any]]:
            import torch

            from megalodon_jax.convert import (
                export_upstream_state_dict,
                load_upstream_state_dict,
            )

            config, jax_model, base_tokens, loss_weights = tiny_reference_fixture()
            torch_state, trainable = make_torch_reference_state(jax_model)
            optimizer = torch.optim.AdamW(
                [torch_state[name] for name in trainable],
                lr=3e-4,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0.01,
                foreach=False,
                fused=False,
            )
            jax_state = export_upstream_state_dict(jax_model)
            first_moment = {name: torch.zeros_like(jax_state[name]) for name in trainable}
            second_moment = {name: torch.zeros_like(jax_state[name]) for name in trainable}
            step_results: list[dict[str, Any]] = []
            passed = True

            for step in range(1, 4):
                token_values = (base_tokens + step - 1) % config.vocab_size
                jax_tokens = jnp.asarray(token_values)
                jax_weights = jnp.asarray(loss_weights)

                def objective(candidate: Any) -> Any:
                    logits, _ = candidate(jax_tokens)
                    return jnp.sum(logits * jax_weights)

                jax_loss, jax_grads = eqx.filter_value_and_grad(objective)(jax_model)
                gradient_state = export_upstream_state_dict(jax_grads)

                optimizer.zero_grad(set_to_none=True)
                torch_logits = source_forward(
                    torch.from_numpy(token_values),
                    torch_state,
                    config,
                )
                torch_loss = torch.sum(torch_logits * torch.from_numpy(loss_weights))
                torch_loss.backward()
                optimizer.step()

                correction1 = 1.0 - 0.9**step
                correction2 = 1.0 - 0.999**step
                with torch.no_grad():
                    for name in trainable:
                        gradient = gradient_state[name]
                        first_moment[name].mul_(0.9).add_(gradient, alpha=0.1)
                        second_moment[name].mul_(0.999).addcmul_(
                            gradient,
                            gradient,
                            value=0.001,
                        )
                        denominator = second_moment[name].sqrt().div_(math.sqrt(correction2))
                        denominator.add_(1e-8)
                        jax_state[name].mul_(1.0 - 3e-4 * 0.01)
                        jax_state[name].addcdiv_(
                            first_moment[name],
                            denominator,
                            value=-3e-4 / correction1,
                        )

                jax_model = load_upstream_state_dict(jax_model, jax_state)
                parameter_errors = {
                    name: float(
                        torch.max(torch.abs(jax_state[name] - torch_state[name].detach())).item()
                    )
                    for name in trainable
                }
                max_parameter_error = max(parameter_errors.values(), default=0.0)
                loss_error = abs(float(jax_loss) - float(torch_loss.detach()))
                step_pass = max_parameter_error <= 3e-6 and loss_error <= 2e-3
                passed = passed and step_pass
                step_results.append({
                    "step": step,
                    "jax_loss": float(jax_loss),
                    "torch_loss": float(torch_loss.detach()),
                    "absolute_loss_error": loss_error,
                    "max_abs_parameter_error": max_parameter_error,
                    "worst_parameter": max(parameter_errors, key=parameter_errors.get),
                    "passed": step_pass,
                })

            return (
                passed,
                "Three AdamW steps preserve source-derived Torch/JAX optimizer parity"
                if passed
                else "Short AdamW parity diverged",
                {
                    "steps": step_results,
                    "parameter_atol": 3e-6,
                    "loss_atol": 2e-3,
                },
            )

        if torch_available:
            audit.run("source_torch_jax_three_step_adamw", "INFO", check_short_adamw_parity)
        else:
            audit.skip(
                "source_torch_jax_three_step_adamw",
                "INFO",
                "PyTorch is unavailable; install the convert extra to run optimizer parity",
                required_extra="convert",
            )

        def check_tiny_overfit() -> tuple[bool, str, dict[str, Any]]:
            details = deterministic_tiny_overfit()
            passed = bool(details.pop("passed"))
            return (
                passed,
                "Deterministic FP32 tiny-batch overfit gate succeeds",
                details,
            )

        audit.run("deterministic_tiny_batch_overfit", "P0", check_tiny_overfit)

        def check_faithful_cache() -> tuple[bool, str, dict[str, Any]]:
            errors = cache_partition_errors(None)
            passed = max(errors.values()) <= 2e-6
            return (
                passed,
                "Faithful chunk-local cache is call-partition invariant"
                if passed
                else "Faithful cache differs between full and tokenwise calls",
                {"max_abs_error": max(errors.values()), "errors": errors},
            )

        audit.run("faithful_cache_partition_invariance", "P1", check_faithful_cache)

        def check_sliding_cache() -> tuple[bool, str, dict[str, Any]]:
            errors = cache_partition_errors(8)
            passed = max(errors.values()) <= 2e-6
            return (
                passed,
                "Sliding cache is call-partition invariant"
                if passed
                else "Sliding-window semantics depend on call granularity",
                {
                    "max_abs_error": max(errors.values()),
                    "errors": errors,
                    "chunk_size": 4,
                    "attention_window": 8,
                },
            )

        audit.run("sliding_cache_partition_invariance", "P1", check_sliding_cache)

    else:
        for name, severity in (
            ("fp32_output_logits", "P1"),
            ("source_torch_jax_forward_gradient_parity", "INFO"),
            ("source_torch_jax_three_step_adamw", "INFO"),
            ("deterministic_tiny_batch_overfit", "P0"),
            ("faithful_cache_partition_invariance", "P1"),
            ("sliding_cache_partition_invariance", "P1"),
        ):
            audit.skip(
                name,
                severity,
                "Not run in fast mode; rerun with --include-slow",
                required_flag="--include-slow",
            )

    manifest_config = tiny_config()
    manifest_model = MegalodonForCausalLM(manifest_config, key=jax.random.PRNGKey(2026))
    parameter_inventory: list[dict[str, Any]] = []

    def upstream_parameter_path(path: str) -> str:
        """Map one canonical native parameter to its exact released-source key."""
        if path == "model.embed.weight":
            return "embed.weight"
        if path == "lm_head.weight":
            return "output.output.weight"
        if path.startswith("model.norm."):
            return f"output.final_norm.{path.removeprefix('model.norm.')}"
        parts = path.split(".")
        if len(parts) < 5 or parts[:2] != ["model", "layers"]:
            raise ValueError(f"unmapped native parameter path: {path}")
        index, family = parts[2], parts[3]
        tail = ".".join(parts[4:])
        if family == "ffn":
            return f"layers.{index}.nffn.{tail}"
        if family != "attn":
            raise ValueError(f"unmapped native parameter family: {path}")
        if tail == "rmsnorm.gamma":
            tail = "rmsnorm.weight"
        elif tail == "cema.gamma_real":
            tail = "cema.gamma[...,0]"
        elif tail == "cema.gamma_imag":
            tail = "cema.gamma[...,1]"
        return f"layers.{index}.mega.{tail}"

    def initializer_contract(path: str) -> str:
        """Describe the initializer for one canonical native parameter."""
        if path in ("model.embed.weight", "lm_head.weight"):
            return "truncated_normal(std=model_dim^-0.5,bounds=[-3sd,3sd])"
        if ".cema.alpha" in path or ".cema.delta" in path:
            return "normal(mean=0,std=0.2)"
        if ".cema.theta" in path:
            return "source_logit_frequency_permutation"
        if ".cema.gamma_real" in path:
            return "normal(mean=0,std=1)"
        if ".cema.omega" in path:
            return "truncated_normal(mean=0,std=0.25,bounds=[-1,1])"
        if path.endswith(".ffn.alpha"):
            return "constant(0.1*0.5^layer_index)"
        if path.endswith(".weight") and any(
            token in path
            for token in (".attn.wz.", ".attn.wv.", ".attn.wr.", ".attn.wh", ".ffn.fc")
        ):
            return str(manifest_config.init_mode)
        return "zeros"

    parameters = model_state_dict(manifest_model)
    for path, leaf in parameters.items():
        shape = tuple(int(size) for size in leaf.shape)
        count = int(np.prod(shape, dtype=np.int64))
        parameter_inventory.append({
            "path": path,
            "shape": shape,
            "dtype": str(leaf.dtype),
            "count": count,
            "trainable": True,
            "initializer": initializer_contract(path),
            "upstream_counterpart": upstream_parameter_path(path),
            "classification": "parameter",
        })
    total_parameters = sum(entry["count"] for entry in parameter_inventory)
    derived_buffers = [
        {
            "path": "model.layers.*.attn.inner.rotary.frequency_schedule",
            "shape_per_layer": (manifest_config.head_dim // 2,),
            "dtype": "float32",
            "count_in_parameter_tree": 0,
            "trainable": False,
            "initializer": "derived(base,head_dim)",
            "upstream_counterpart": "rope.freqs",
            "classification": "derived_buffer",
        }
    ]
    upstream_config_fields = {
        "vocab_size",
        "model_dim",
        "num_layers",
        "num_heads",
        "z_dim",
        "value_dim",
        "ffn_hidden_dim",
        "cema_ndim",
        "chunk_size",
        "norm_num_groups",
        "norm_eps",
        "rope_base",
        "swiglu",
        "rescale_nffn",
        "scale_emb",
        "share_emb",
        "norm_affine",
        "dropout",
        "attention_dropout",
        "hidden_dropout",
        "output_size",
        "init_mode",
    }
    static_values = [
        {
            "path": f"config.{field.name}",
            "value": _jsonable(getattr(manifest_config, field.name)),
            "dtype": type(getattr(manifest_config, field.name)).__name__,
            "trainable": False,
            "initializer": None,
            "upstream_counterpart": (
                f"ModelConf.{field.name}"
                if field.name in upstream_config_fields
                else "none (JAX-specific static contract)"
            ),
            "classification": "static_value",
        }
        for field in dataclasses.fields(manifest_config)
    ]

    passed = sum(result.status == "passed" for result in audit.results)
    failed = sum(result.status == "failed" for result in audit.results)
    skipped = sum(result.status == "skipped" for result in audit.results)

    print("\nMEGALODON JAX MODELING CORRECTNESS VERIFICATION")
    print(f"Repository: {repo}")
    print(f"Upstream evidence: {upstream if upstream_available else 'unavailable'}")
    print(f"JAX: {jax.__version__}; platform: {jax.default_backend()}")
    print("-" * 100)
    for result in audit.results:
        mark = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP"}[result.status]
        print(f"{mark:4s}  {result.severity:2s}  {result.name:42s}  {result.summary}")
        if result.status == "failed":
            # Keep console concise; JSON retains complete tensors/details.
            compact = {
                key: value
                for key, value in result.details.items()
                if key not in {"actual_output", "expected_output", "traceback"}
            }
            print(f"      details: {json.dumps(compact, sort_keys=True)}")
    print("-" * 100)
    print(
        f"Summary: {passed} passed, {failed} failed, {skipped} skipped, "
        f"{len(audit.results)} declared"
    )
    if skipped:
        print(f"{skipped} checks skipped; inspect SKIP rows and rerun with required flags")

    report = {
        "repository": str(repo),
        "repository_commit": _git_revision(repo),
        "upstream_repository": str(upstream) if upstream_available else None,
        "upstream_config_sha256": (
            _sha256(upstream_config_path) if upstream_config_path.is_file() else None
        ),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "equinox_version": importlib.metadata.version("equinox"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch": _torch_environment(),
        "parameter_manifest": {
            "config": _jsonable(dataclasses.asdict(manifest_config)),
            "total_trainable_elements": total_parameters,
            "formula_total": manifest_config.parameter_count_breakdown()["total"],
            "parameters": parameter_inventory,
            "derived_buffers": derived_buffers,
            "static_values": static_values,
        },
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "results": [result.as_dict() for result in audit.results],
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"JSON report: {args.json.resolve()}")

    if failed and not args.no_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
