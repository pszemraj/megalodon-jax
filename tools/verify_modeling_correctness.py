#!/usr/bin/env python3
"""Verify Megalodon JAX modeling semantics against local authoritative sources.

This standalone verifier compares the JAX implementation against independent mathematical
oracles and the exact local original PyTorch/CUDA source. It uses small deterministic
examples and does not build the fused CUDA extension, import downstream ports, or import
an installed Megalodon package.

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
from typing import Any

# These must be set before importing JAX.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
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
                passed=bool(passed),
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
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


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


def current_jax_formula_count(config: Any, *, assume_tied: bool) -> int:
    """Count the current JAX architecture formula without allocating a 7B model."""

    d = int(config.model_dim)
    layers = int(config.num_layers)
    h = int(config.num_heads)
    z = int(config.z_dim)
    v = int(config.value_dim)
    f = int(config.ffn_hidden_dim)
    n = int(config.cema_ndim)
    vocab = int(config.vocab_size)
    affine = bool(config.norm_affine)

    timestep_norm = 2 * d if affine else 0
    cema = 4 * d * n + 2 * d
    rmsnorm = d if affine else 0
    projections = (
        (d * z + z) + (d * v + v) + (d * v + v) + (d * d + d) + (v * d + d)  # current JAX wh2 bias
    )
    qk_affine = 4 * z
    # Current inv_freq is an inexact array leaf and is trainable under generic filters.
    rope_leaf = (z // h) // 2
    ffn_norm = 2 * d if affine else 0
    ffn = (d * f + f) + (f * d + d) + ((d * f + f) if config.swiglu else 0)
    per_layer = (
        timestep_norm + cema + rmsnorm + projections + qk_affine + rope_leaf + ffn_norm + ffn
    )
    final_norm = 2 * d if affine else 0
    embedding = vocab * d
    output = 0 if assume_tied else vocab * d
    return embedding + layers * per_layer + final_norm + output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path("/mnt/data/megalodon-jax-extract/megalodon-jax")
    parser.add_argument(
        "--jax-repo",
        type=Path,
        default=default_repo if default_repo.exists() else Path.cwd(),
        help="Root of the megalodon-jax repository (must contain src/megalodon_jax).",
    )
    parser.add_argument(
        "--upstream-repo",
        type=Path,
        default=None,
        help="Exact local original PyTorch/CUDA repository (defaults under local-scratch).",
    )
    parser.add_argument(
        "--paper",
        type=Path,
        default=None,
        help="Local paper Markdown path (defaults under local-scratch).",
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
    paper = (
        args.paper.resolve()
        if args.paper is not None
        else repo / "local-scratch" / "megalodon_paper.md"
    )
    src = repo / "src"
    package = src / "megalodon_jax"
    if not package.is_dir():
        print(f"ERROR: {package} does not exist", file=sys.stderr)
        return 2
    if not upstream.is_dir():
        print(f"ERROR: local upstream repository does not exist: {upstream}", file=sys.stderr)
        return 2
    if not paper.is_file():
        print(f"ERROR: local paper does not exist: {paper}", file=sys.stderr)
        return 2
    sys.path.insert(0, str(src))

    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from megalodon_jax.config import MegalodonConfig
    from megalodon_jax.layers import RotaryEmbedding, TimestepNorm
    from megalodon_jax.layers.attention import ChunkedAttention
    from megalodon_jax.layers.norms import BatchedLayerNorm
    from megalodon_jax.model import MegalodonForCausalLM
    from megalodon_jax.utils import get_initializer

    audit = Audit()

    def check_timestep_math() -> tuple[bool, str, dict[str, Any]]:
        x = np.array([[[1.0, 3.0, 2.0, 6.0], [5.0, 7.0, 10.0, 14.0]]], dtype=np.float32)
        module = TimestepNorm(4, 2, eps=0.0)
        actual, state = module(jnp.asarray(x))
        expected, count, mean, var = exact_timestep_norm(x, 2, eps=0.0)
        y_err = float(np.max(np.abs(np.asarray(actual) - expected)))
        var_err = float(np.max(np.abs(np.asarray(state.var) - var)))
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
        module = TimestepNorm(4, 2, eps=0.0)
        actual, state = module(jnp.asarray(x), mask=jnp.asarray(mask))
        expected, count, mean, var = exact_timestep_norm(x, 2, mask=mask, eps=0.0)
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
        module = TimestepNorm(4, 2, eps=0.0)
        actual, state = module(jnp.asarray(x), segment_ids=jnp.asarray(segments))
        expected, count, mean, var = exact_timestep_norm(x, 2, segment_ids=segments, eps=0.0)
        err = float(np.max(np.abs(np.asarray(actual) - expected)))
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
        q_err = float(np.max(np.abs(np.asarray(q_actual) - q_expected)))
        k_err = float(np.max(np.abs(np.asarray(k_actual) - k_expected)))
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
        q = jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 1, 1, 8)
        k = q * 1.3

        def objective(m: Any) -> Any:
            qo, ko = m(q, k, jnp.asarray(7, jnp.int32))
            return jnp.sum(qo * 0.37 + ko * 0.11)

        grad = eqx.filter_grad(objective)(module)
        inv_grad = getattr(grad, "inv_freq", None)
        if inv_grad is None:
            norm = 0.0
            passed = True
        else:
            norm = float(jnp.linalg.norm(inv_grad))
            passed = norm == 0.0
        return (
            passed,
            "RoPE frequencies are non-trainable"
            if passed
            else "RoPE inv_freq receives gradients and will enter generic Equinox optimizers",
            {"inv_freq_gradient": inv_grad, "gradient_l2_norm": norm},
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
        effective_identity = float(np.max(np.abs(y - identity_expected))) <= 2e-6
        # Source-compatible storage must be zero while effective scale is one.
        passed = np.array_equal(stored, np.zeros_like(stored)) and effective_identity
        return (
            passed,
            "LayerNorm stores zero and applies weight+1"
            if passed
            else "LayerNorm uses direct one-scale storage; upstream checkpoint zeros would collapse output",
            {
                "stored_weight": stored,
                "effective_identity_error": float(np.max(np.abs(y - identity_expected))),
            },
        )

    audit.run("layernorm_plus_one_storage", "P0", check_layernorm_storage)

    def check_explicit_tying() -> tuple[bool, str, dict[str, Any]]:
        fields = {field.name for field in dataclasses.fields(MegalodonConfig)}
        has_flag = "share_emb" in fields
        cfg = MegalodonConfig(
            vocab_size=32,
            output_size=32,
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
        cfg = MegalodonConfig.from_7b()
        matches_71 = cfg.ffn_hidden_dim == 11264 and not cfg.swiglu
        matches_73 = cfg.ffn_hidden_dim == 8192 and cfg.swiglu
        source_71 = source_parameter_count(
            model_dim=4096,
            num_layers=32,
            num_heads=4,
            z_dim=1024,
            value_dim=8192,
            ffn_hidden_dim=11264,
            cema_ndim=16,
            vocab_size=32000,
            swiglu=False,
            share_emb=False,
        )
        source_73 = source_parameter_count(
            model_dim=4096,
            num_layers=32,
            num_heads=4,
            z_dim=1024,
            value_dim=8192,
            ffn_hidden_dim=8192,
            cema_ndim=16,
            vocab_size=32000,
            swiglu=True,
            share_emb=False,
        )
        current_tied = current_jax_formula_count(cfg, assume_tied=True)
        passed = matches_71 or matches_73
        return (
            passed,
            "7B factory matches a named upstream preset"
            if passed
            else "from_7b combines F=11264 with SwiGLU and is an ~8.46B hybrid",
            {
                "factory_ffn_hidden_dim": cfg.ffn_hidden_dim,
                "factory_swiglu": cfg.swiglu,
                "factory_init_mode": cfg.init_mode,
                "current_formula_tied_count": current_tied,
                "upstream_mega7_1b_count": source_71,
                "upstream_mega7_3b_count": source_73,
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
            "wh1": (1024, 1024),
            "wh2": (1024, 2048),
            "fc1": (2560, 1024),
            "fc2": (1024, 2560),
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

    audit.run("upstream_initialization_contract", "INFO", check_upstream_init_contract)

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

    audit.run("cuda_timestep_mask_source_contract", "INFO", check_cuda_source_mask_contract)

    def check_source_biases() -> tuple[bool, str, dict[str, Any]]:
        cfg = MegalodonConfig(
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
            swiglu=True,
        )
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

    def check_pad_embedding() -> tuple[bool, str, dict[str, Any]]:
        import inspect

        # The uploaded upstream model performs a normal embedding lookup and has no
        # padding_idx. The JAX forward explicitly replaces pad-token embeddings with
        # zero. With the bundled tokenizer, pad_token == unk_token == ID 0, so this
        # also removes the unknown-token embedding and its gradient.
        from megalodon_jax.model import MegalodonModel

        source = inspect.getsource(MegalodonModel.__call__)
        forced_zero_mask = (
            "pad_mask = input_ids == self.config.pad_token_id" in source
            and "jnp.where(pad_mask" in source
        )
        passed = not forced_zero_mask
        return (
            passed,
            "Embedding lookup is not forcibly zero-masked by token ID"
            if passed
            else "Forward zero-masks pad_token_id; bundled ID 0 is also <unk>",
            {
                "forced_pad_zero_mask_detected": forced_zero_mask,
                "default_pad_token_id": MegalodonConfig().pad_token_id,
            },
        )

    audit.run("upstream_embedding_padding_semantics", "P1", check_pad_embedding)

    if args.include_slow:

        def check_logits_dtype() -> tuple[bool, str, dict[str, Any]]:
            cfg = MegalodonConfig(
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

        def cache_partition_error(cache_size: int, cache_unbounded: bool) -> float:
            key = jax.random.PRNGKey(17)
            k_module, kq, kk, kv = jax.random.split(key, 4)
            batch, length, heads, dim, value_dim = 1, 12, 1, 4, 3
            module = ChunkedAttention(
                num_heads=heads,
                head_dim=dim,
                value_head_dim=value_dim,
                chunk_size=4,
                max_cache_len=cache_size,
                cache_unbounded=cache_unbounded,
                key=k_module,
            )
            q = jax.random.normal(kq, (batch, length, heads, dim))
            k = jax.random.normal(kk, (batch, length, heads, dim))
            v = jax.random.normal(kv, (batch, length, heads, value_dim))
            full, _, _ = module(q, k, v, return_cache=True)
            pieces = []
            cache = None
            for index in range(length):
                part, cache, _ = module(
                    q[:, index : index + 1],
                    k[:, index : index + 1],
                    v[:, index : index + 1],
                    cache=cache,
                    return_cache=True,
                )
                pieces.append(part)
            tokenwise = jnp.concatenate(pieces, axis=1)
            return float(jnp.max(jnp.abs(full - tokenwise)))

        def check_faithful_cache() -> tuple[bool, str, dict[str, Any]]:
            error = cache_partition_error(4, False)
            passed = error <= 2e-6
            return (
                passed,
                "Faithful chunk-local cache is call-partition invariant"
                if passed
                else "Faithful cache differs between full and tokenwise calls",
                {"max_abs_error": error},
            )

        audit.run("faithful_cache_partition_invariance", "P1", check_faithful_cache)

        def check_sliding_cache() -> tuple[bool, str, dict[str, Any]]:
            error = cache_partition_error(8, False)
            passed = error <= 2e-6
            return (
                passed,
                "Sliding cache is call-partition invariant"
                if passed
                else "Sliding-window semantics depend on call granularity",
                {"max_abs_error": error, "chunk_size": 4, "max_cache_len": 8},
            )

        audit.run("sliding_cache_partition_invariance", "P1", check_sliding_cache)

    manifest_config = MegalodonConfig(
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
    manifest_model = MegalodonForCausalLM(manifest_config, key=jax.random.PRNGKey(2026))
    parameter_inventory = []
    total_parameters = 0
    for path, leaf in jax.tree_util.tree_flatten_with_path(manifest_model)[0]:
        if not eqx.is_array(leaf):
            continue
        shape = tuple(int(size) for size in leaf.shape)
        count = int(np.prod(shape, dtype=np.int64))
        total_parameters += count
        path_text = jax.tree_util.keystr(path)
        trainable = bool(jnp.issubdtype(leaf.dtype, jnp.inexact))
        if ".embed.weight" in path_text or ".lm_head.weight" in path_text:
            initializer = "boundary_truncated_normal"
        elif any(token in path_text for token in ("weight", "wz", "wv", "wr", "wh", "fc")):
            initializer = str(manifest_config.init_mode)
        elif trainable:
            initializer = "module_specific"
        else:
            initializer = "derived_or_state"
        parameter_inventory.append(
            {
                "path": path_text,
                "shape": shape,
                "dtype": str(leaf.dtype),
                "count": count,
                "trainable": trainable,
                "initializer": initializer,
                "upstream_counterpart": path_text,
                "classification": "parameter" if trainable else "derived_buffer",
            }
        )

    passed = sum(result.passed for result in audit.results)
    failed = len(audit.results) - passed

    print("\nMEGALODON JAX MODELING CORRECTNESS VERIFICATION")
    print(f"Repository: {repo}")
    print(f"Upstream evidence: {upstream}")
    print(f"Paper evidence: {paper}")
    print(f"JAX: {jax.__version__}; platform: {jax.default_backend()}")
    print("-" * 100)
    for result in audit.results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"{mark:4s}  {result.severity:2s}  {result.name:42s}  {result.summary}")
        if not result.passed:
            # Keep console concise; JSON retains complete tensors/details.
            compact = {
                key: value
                for key, value in result.details.items()
                if key not in {"actual_output", "expected_output", "traceback"}
            }
            print(f"      details: {json.dumps(compact, sort_keys=True)}")
    print("-" * 100)
    print(f"Summary: {passed} passed, {failed} failed, {len(audit.results)} total")

    report = {
        "repository": str(repo),
        "repository_commit": _git_revision(repo),
        "upstream_repository": str(upstream),
        "paper": str(paper),
        "paper_sha256": _sha256(paper),
        "upstream_config_sha256": _sha256(upstream / "megalodon" / "config.py"),
        "python_version": platform.python_version(),
        "jax_version": jax.__version__,
        "equinox_version": importlib.metadata.version("equinox"),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch": _torch_environment(),
        "parameter_manifest": {
            "config": _jsonable(dataclasses.asdict(manifest_config)),
            "total_array_elements": total_parameters,
            "leaves": parameter_inventory,
        },
        "passed": passed,
        "failed": failed,
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
