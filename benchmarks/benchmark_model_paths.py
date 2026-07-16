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
"""Benchmark production inference and training paths with auditable evidence.

The default process is a supervisor. It expands the requested matrix and runs
each case in a fresh worker subprocess, which prevents one XLA crash, timeout,
or OOM from destroying the rest of a long benchmark. Workers import
``megalodon_jax`` from the requested repository root, so clean worktrees for
the current branch and historical ``main`` can be measured with the exact same
driver and environment.

The canonical cross-revision model is a reduced, stable comparison topology,
not a paper-7B feasibility test. It uses model dimension 1024, 12 layers, one
attention head, vocabulary 16,000, and tied embedding/output weights so its
parameter topology remains comparable with historical revisions that inferred
tying from output shape. Pass ``--config-json '{"share_emb": false}'`` to
measure the current untied production topology separately. Configuration
fields are selected by dataclass introspection, allowing the same request to
target both the current API and older checkouts whose precision or cache field
names differ.

Examples::

    # Current checkout, full default matrix.
    conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
        --repo current=. \
        --output local-scratch/model-paths-current.json

    # Compare clean current/main worktrees with the default XLA configuration.
    conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
        --repo current=. --repo main=/tmp/megalodon-jax-main \
        --suite inference --inference-lengths 64,512,2048,4096 \
        --output local-scratch/model-paths-current-vs-main.json

    # Small smoke run. The normal model remains canonical unless overridden.
    conda run --name mega-jax python benchmarks/benchmark_model_paths.py \
        --repo current=. --suite inference --inference-operations noncached \
        --inference-lengths 8 --warmups 0 --iterations 1 --allow-dirty \
        --config-json '{"vocab_size":128,"model_dim":32,"num_layers":1,\
"num_heads":1,"z_dim":16,"value_dim":32,"ffn_hidden_dim":64,\
"cema_ndim":2,"chunk_size":8,"norm_num_groups":4}' \
        --output /tmp/megalodon-jax-model-path-smoke.json

Compilation is excluded from runtime samples and reported separately. Inputs
are placed on device before lowering; every warmup and timed invocation is
synchronized. The driver contains no FFI, custom calls, Pallas, or Triton code.
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
import re
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

Status = Literal["passed", "completed_noncomparable", "oom", "timeout", "failed"]

CANONICAL_CONFIG: dict[str, Any] = {
    "vocab_size": 16_000,
    "model_dim": 1024,
    "num_layers": 12,
    "num_heads": 1,
    "z_dim": 256,
    "value_dim": 2048,
    "ffn_hidden_dim": 2560,
    "cema_ndim": 16,
    "chunk_size": 2048,
    "attention_window": None,
    "norm_num_groups": 32,
    "norm_eps": 1e-5,
    "rope_base": None,
    "swiglu": False,
    "rescale_nffn": False,
    "scale_emb": False,
    "share_emb": True,
    "norm_affine": True,
    "dropout": 0.0,
    "attention_dropout": 0.0,
    "attention_dropout_mode": "post_softmax",
    "hidden_dropout": 0.0,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "init_mode": "he",
    "use_associative_segment_scan": True,
    "output_size": -1,
    "param_dtype": "float32",
    "compute_dtype": "bfloat16",
    "accum_dtype": "float32",
    "attention_softmax_dtype": "float32",
    "loss_softmax_dtype": "float32",
}

CUDA_PACKAGE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "jax_cuda13_plugin": ("jax-cuda13-plugin",),
    "jax_cuda13_pjrt": ("jax-cuda13-pjrt",),
    "cublas": ("nvidia-cublas-cu13", "nvidia-cublas"),
    "cuda_runtime": ("nvidia-cuda-runtime-cu13", "nvidia-cuda-runtime"),
    "cudnn": ("nvidia-cudnn-cu13", "nvidia-cudnn"),
    "cufft": ("nvidia-cufft-cu13", "nvidia-cufft"),
}

INFERENCE_OPERATIONS = (
    "noncached",
    "pristine_prefill",
    "continuation_37",
    "decode_1",
    "lm_ttft",
)
TRAINING_OPERATIONS = ("forward", "forward_backward")
TRAINING_MODES = ("plain", "all_true", "packed")


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    """Atomically replace ``path`` with a formatted JSON document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    """Convert common scientific-Python values into strict JSON values."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        try:
            return _jsonable(value.tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _run_git(repo: Path, *arguments: str) -> str | None:
    """Run a read-only Git command, returning ``None`` when unavailable."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _repo_provenance(repo: Path) -> dict[str, Any]:
    """Describe an exact source tree without accepting an enclosing checkout."""
    repo = repo.resolve()
    top_level = _run_git(repo, "rev-parse", "--show-toplevel")
    is_checkout_root = top_level is not None and Path(top_level).resolve() == repo
    revision = _run_git(repo, "rev-parse", "HEAD") if is_checkout_root else None
    status = _run_git(repo, "status", "--porcelain=v1") if is_checkout_root else None
    branch = _run_git(repo, "branch", "--show-current") if is_checkout_root else None
    discovered = {
        path.relative_to(repo).as_posix()
        for path in (repo / "src/megalodon_jax").rglob("*.py")
        if path.is_file()
    }
    if (repo / "pyproject.toml").is_file():
        discovered.add("pyproject.toml")
    files: dict[str, Any] = {}
    tree_digest = hashlib.sha256()
    for relative in sorted(discovered):
        source = repo / relative
        file_digest = _sha256(source)
        files[relative] = {
            "present": source.is_file(),
            "sha256": file_digest,
        }
        tree_digest.update(relative.encode("utf-8"))
        tree_digest.update(b"\0")
        tree_digest.update(file_digest.encode("ascii"))
        tree_digest.update(b"\n")
    return {
        "root": str(repo),
        "git_checkout_root": is_checkout_root,
        "revision": revision or os.environ.get("MEGALODON_JAX_SOURCE_REVISION", "unknown"),
        "branch": branch or None,
        "dirty": bool(status) if status is not None else None,
        "dirty_entries": status.splitlines() if status else [],
        "source_tree_sha256": tree_digest.hexdigest(),
        "source_files": files,
    }


def _parse_named_repo(value: str) -> tuple[str, Path]:
    """Parse ``NAME=PATH`` or infer a name from a bare path."""
    if "=" in value:
        name, raw_path = value.split("=", 1)
    else:
        raw_path = value
        name = Path(value).resolve().name
    if not name or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise argparse.ArgumentTypeError(f"invalid repository label: {name!r}")
    path = Path(raw_path).expanduser().resolve()
    if not (path / "src/megalodon_jax").is_dir():
        raise argparse.ArgumentTypeError(f"not a megalodon-jax source root: {path}")
    return name, path


def _csv_strings(value: str) -> tuple[str, ...]:
    """Parse a nonempty comma-separated string list."""
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _csv_positive_ints(value: str) -> tuple[int, ...]:
    """Parse a comma-separated positive integer list."""
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid integer list: {value!r}") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all list values must be positive integers")
    return values


def _parse_config_json(value: str | None) -> dict[str, Any]:
    """Parse inline JSON or ``@path`` configuration overrides."""
    if value is None:
        return {}
    if value.startswith("@"):
        text = Path(value[1:]).expanduser().read_text(encoding="utf-8")
    else:
        text = value
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("--config-json must contain a JSON object")
    return parsed


def _case_id(case: Mapping[str, Any]) -> str:
    """Return a filesystem-safe stable identifier for one benchmark case."""
    fields = [
        case["repo_name"],
        case["suite"],
        case["operation"],
        case.get("mode", "none"),
        f"b{case['batch_size']}",
        f"l{case['sequence_length']}",
    ]
    return "--".join(str(field).replace("_", "-") for field in fields)


def _case_profile_dir(case: Mapping[str, Any]) -> Path | None:
    """Return the isolated XProf output directory requested for one case."""
    root = case.get("profile_root")
    if root is None:
        return None
    return Path(str(root)) / str(case["case_id"])


def _build_cases(
    args: argparse.Namespace, repos: Sequence[tuple[str, Path]]
) -> list[dict[str, Any]]:
    """Expand command-line matrices into isolated worker specifications."""
    suites = tuple(args.suite or ("inference", "training"))
    invalid_inference = set(args.inference_operations) - set(INFERENCE_OPERATIONS)
    invalid_training = set(args.training_operations) - set(TRAINING_OPERATIONS)
    invalid_modes = set(args.training_modes) - set(TRAINING_MODES)
    if invalid_inference:
        raise ValueError(f"unsupported inference operations: {sorted(invalid_inference)}")
    if invalid_training:
        raise ValueError(f"unsupported training operations: {sorted(invalid_training)}")
    if invalid_modes:
        raise ValueError(f"unsupported training modes: {sorted(invalid_modes)}")

    config_overrides = _parse_config_json(args.config_json)
    compute_dtype = str(config_overrides.get("compute_dtype", CANONICAL_CONFIG["compute_dtype"]))
    if "bfloat16" in compute_dtype.lower():
        default_atol = args.bf16_atol
        default_rtol = args.bf16_rtol
        tolerance_policy = "bfloat16"
    else:
        default_atol = args.fp32_atol
        default_rtol = args.fp32_rtol
        tolerance_policy = "float32"
    common = {
        "warmups": args.warmups,
        "iterations": args.iterations,
        "seed": args.seed,
        "atol": default_atol if args.atol is None else args.atol,
        "rtol": default_rtol if args.rtol is None else args.rtol,
        "tolerance_policy": tolerance_policy
        if args.atol is None and args.rtol is None
        else "explicit",
        "config_overrides": config_overrides,
        "loss_chunk_size": args.loss_chunk_size,
        "profile_root": str(args.profile_dir.expanduser().resolve())
        if args.profile_dir is not None
        else None,
    }
    cases: list[dict[str, Any]] = []
    for repo_name, repo_root in repos:
        if "inference" in suites:
            for batch in args.inference_batches:
                for length in args.inference_lengths:
                    for operation in args.inference_operations:
                        case = {
                            **common,
                            "repo_name": repo_name,
                            "repo_root": str(repo_root),
                            "suite": "inference",
                            "operation": operation,
                            "mode": "inference",
                            "batch_size": batch,
                            "sequence_length": length,
                        }
                        case["case_id"] = _case_id(case)
                        cases.append(case)
        if "training" in suites:
            for batch in args.training_batches:
                for length in args.training_lengths:
                    for mode in args.training_modes:
                        for operation in args.training_operations:
                            case = {
                                **common,
                                "repo_name": repo_name,
                                "repo_root": str(repo_root),
                                "suite": "training",
                                "operation": operation,
                                "mode": mode,
                                "batch_size": batch,
                                "sequence_length": length,
                            }
                            case["case_id"] = _case_id(case)
                            cases.append(case)
    return cases


def _classify_failure(text: str, returncode: int | None = None) -> Status:
    """Classify a worker failure without relying on one backend exception type."""
    lowered = text.lower()
    oom_markers = (
        "out of memory",
        "resource_exhausted",
        "cuda_error_out_of_memory",
        "failed to allocate",
        "allocator ran out",
    )
    if any(marker in lowered for marker in oom_markers) or returncode in (-9, 137):
        return "oom"
    return "failed"


def _supervisor(args: argparse.Namespace) -> int:
    """Run each planned case in an isolated subprocess and aggregate evidence."""
    repos = [_parse_named_repo(value) for value in (args.repo or ["current=."])]
    if len({name for name, _ in repos}) != len(repos):
        raise ValueError("repository labels must be unique")
    provenances = {name: _repo_provenance(path) for name, path in repos}
    for name, provenance in provenances.items():
        if provenance["revision"] == "unknown" and not args.allow_unknown_revision:
            raise RuntimeError(
                f"{name}: exact Git revision unavailable; use a clean Git worktree or "
                "--allow-unknown-revision"
            )
        if provenance["dirty"] is True and not args.allow_dirty:
            entries = "\n".join(provenance["dirty_entries"][:20])
            raise RuntimeError(
                f"{name}: source tree is dirty; commit/stash changes or pass --allow-dirty:\n{entries}"
            )

    cases = _build_cases(args, repos)
    output = args.output.expanduser().resolve()
    case_directory = output.parent / f"{output.stem}.cases"
    script = Path(__file__).resolve()
    run: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": "megalodon-jax-production-model-paths",
        "status": "planned" if args.dry_run else "running",
        "started_at": _utc_now(),
        "finished_at": None,
        "benchmark_script": {"path": str(script), "sha256": _sha256(script)},
        "supervisor": {
            "python": sys.version,
            "platform": platform.platform(),
            "argv": sys.argv,
            "timeout_seconds": args.timeout_seconds,
            "warmups": args.warmups,
            "iterations": args.iterations,
            "profile_directory": str(args.profile_dir.expanduser().resolve())
            if args.profile_dir is not None
            else None,
        },
        "repositories": provenances,
        "planned_cases": len(cases),
        "completed_cases": 0,
        "summary": {
            status: 0
            for status in ("passed", "completed_noncomparable", "oom", "timeout", "failed")
        },
        "cases": [],
        "case_plan": cases if args.dry_run else None,
    }
    _atomic_json(output, run)
    if args.dry_run:
        run["status"] = "planned"
        run["finished_at"] = _utc_now()
        _atomic_json(output, run)
        print(f"planned {len(cases)} cases -> {output}")
        return 0

    case_directory.mkdir(parents=True, exist_ok=True)
    interrupted = False
    try:
        for index, case in enumerate(cases, start=1):
            case_id = case["case_id"]
            spec_path = case_directory / f"{case_id}.spec.json"
            result_path = case_directory / f"{case_id}.json"
            _atomic_json(spec_path, case)
            result_path.unlink(missing_ok=True)
            command = [
                sys.executable,
                str(script),
                "--_worker-spec",
                str(spec_path),
                "--_worker-result",
                str(result_path),
            ]
            print(f"[{index}/{len(cases)}] {case_id}", flush=True)
            started = time.perf_counter()
            try:
                process = subprocess.run(
                    command,
                    cwd=case["repo_root"],
                    capture_output=True,
                    text=True,
                    timeout=args.timeout_seconds,
                    check=False,
                )
                duration = time.perf_counter() - started
                if result_path.is_file():
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                else:
                    combined = f"{process.stdout}\n{process.stderr}"
                    result = {
                        **case,
                        "status": _classify_failure(combined, process.returncode),
                        "started_at": None,
                        "finished_at": _utc_now(),
                        "worker_wall_seconds": duration,
                        "error": {
                            "type": "WorkerProcessError",
                            "message": f"worker exited {process.returncode} without a result",
                        },
                    }
                result["worker_process"] = {
                    "returncode": process.returncode,
                    "wall_seconds": duration,
                    "stdout_tail": process.stdout[-20_000:],
                    "stderr_tail": process.stderr[-20_000:],
                }
            except subprocess.TimeoutExpired as error:
                duration = time.perf_counter() - started
                result = {
                    **case,
                    "status": "timeout",
                    "started_at": None,
                    "finished_at": _utc_now(),
                    "worker_wall_seconds": duration,
                    "error": {
                        "type": "TimeoutExpired",
                        "message": f"worker exceeded {args.timeout_seconds} seconds",
                    },
                    "worker_process": {
                        "returncode": None,
                        "wall_seconds": duration,
                        "stdout_tail": (error.stdout or "")[-20_000:],
                        "stderr_tail": (error.stderr or "")[-20_000:],
                    },
                }
                _atomic_json(result_path, result)

            status = result.get("status", "failed")
            if status not in run["summary"]:
                status = "failed"
                result["status"] = status
            run["summary"][status] += 1
            run["cases"].append(result)
            run["completed_cases"] = len(run["cases"])
            _atomic_json(result_path, result)
            _atomic_json(output, run)
            print(f"    {status}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        run["finished_at"] = _utc_now()
        if interrupted:
            run["status"] = "interrupted"
        elif run["summary"]["failed"] or run["summary"]["oom"] or run["summary"]["timeout"]:
            run["status"] = "completed_with_failures"
        elif run["summary"]["completed_noncomparable"]:
            run["status"] = "completed_with_noncomparable"
        else:
            run["status"] = "passed"
        _atomic_json(output, run)

    print(
        f"{run['status']}: {run['completed_cases']}/{run['planned_cases']} cases; "
        f"{run['summary']} -> {output}"
    )
    if interrupted:
        return 130
    has_failures = any(run["summary"][name] for name in ("failed", "oom", "timeout"))
    return 0 if args.allow_failures or not has_failures else 1


def _block_tree(jax: Any, tree: Any) -> None:
    """Synchronize every device array in a pytree."""
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _timing_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Summarize synchronized samples while retaining every observation."""
    import numpy as np

    return {
        "iterations": len(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "p90_ms": float(np.percentile(samples_ms, 90)),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": list(samples_ms),
    }


def _measure(
    jax: Any,
    compiled: Any,
    arguments: tuple[Any, ...],
    warmups: int,
    iterations: int,
    profile_dir: Path | None = None,
) -> tuple[dict[str, Any], Any]:
    """Measure one compiled executable with explicit synchronization."""
    for _ in range(warmups):
        _block_tree(jax, compiled(*arguments))
    samples: list[float] = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter_ns()
        result = compiled(*arguments)
        _block_tree(jax, result)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    summary = _timing_summary(samples)
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        with jax.profiler.trace(profile_dir, create_perfetto_link=False):
            profile_result = compiled(*arguments)
            _block_tree(jax, profile_result)
        summary["xprof_trace"] = {
            "directory": str(profile_dir),
            "timing_excluded": True,
            "profiled_iterations": 1,
        }
    return summary, result


def _public_attributes(value: Any) -> dict[str, Any]:
    """Extract scalar fields from backend analysis objects."""
    attributes: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except Exception:
            continue
        if callable(item):
            continue
        if item is None or isinstance(item, (bool, int, float, str)) or hasattr(item, "item"):
            attributes[name] = _jsonable(item)
    return attributes


def _compiler_metadata(lowered: Any, compiled: Any) -> dict[str, Any]:
    """Collect StableHLO, cost, and memory evidence across JAX versions."""
    # Equinox's filtered stages wrap the public JAX stages so that static
    # arguments can be reconstructed around execution. Compiler inspection
    # belongs to the wrapped JAX objects on Equinox 0.13.x.
    jax_lowered = getattr(lowered, "lowered", lowered)
    jax_compiled = getattr(compiled, "compiled", compiled)
    try:
        stablehlo = str(jax_lowered.compiler_ir(dialect="stablehlo"))
        stablehlo_record: dict[str, Any] = {
            "available": True,
            "bytes": len(stablehlo.encode("utf-8")),
            "while_count": stablehlo.count("stablehlo.while"),
            "custom_call_count": stablehlo.count("stablehlo.custom_call"),
            "dynamic_update_slice_count": stablehlo.count("stablehlo.dynamic_update_slice"),
        }
    except Exception as error:
        stablehlo_record = {
            "available": False,
            "error": f"{type(error).__name__}: {error}",
        }

    cost: dict[str, Any]
    try:
        analysis = jax_lowered.cost_analysis()
        cost = {"available": True, "analysis": _jsonable(analysis)}
    except Exception as error:
        try:
            analysis = jax_compiled.cost_analysis()
            cost = {"available": True, "analysis": _jsonable(analysis)}
        except Exception as compiled_error:
            cost = {
                "available": False,
                "error": (
                    f"lowered={type(error).__name__}: {error}; "
                    f"compiled={type(compiled_error).__name__}: {compiled_error}"
                ),
            }

    try:
        analysis = jax_compiled.memory_analysis()
        if analysis is None:
            memory = {"available": False, "error": "backend returned no memory analysis"}
        else:
            memory = {"available": True, **_public_attributes(analysis)}
    except Exception as error:
        memory = {"available": False, "error": f"{type(error).__name__}: {error}"}
    return {"stablehlo": stablehlo_record, "cost_analysis": cost, "memory_analysis": memory}


def _installed_package_versions(
    candidates: Mapping[str, Sequence[str]] = CUDA_PACKAGE_CANDIDATES,
) -> dict[str, Any]:
    """Resolve CUDA packages across suffixed and unsuffixed wheel names."""
    records: dict[str, Any] = {}
    for component, distributions in candidates.items():
        installed = []
        for distribution in distributions:
            try:
                version = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                continue
            installed.append({"distribution": distribution, "version": version})
        records[component] = {
            "installed": installed,
            "searched_distributions": list(distributions),
        }
    return records


def _command_provenance(command: Sequence[str]) -> dict[str, Any]:
    """Run a read-only environment probe without making benchmarks depend on it."""
    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {
            "available": False,
            "command": list(command),
            "error": f"{type(error).__name__}: {error}",
        }
    return {
        "available": result.returncode == 0,
        "command": list(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _nvidia_driver_version(nvidia_smi: Mapping[str, Any]) -> str | None:
    """Extract the NVIDIA driver version from full ``nvidia-smi`` output."""
    if not nvidia_smi.get("available"):
        return None
    match = re.search(r"Driver Version:\s*([^\s|]+)", str(nvidia_smi.get("stdout", "")))
    return match.group(1) if match is not None else None


def _device_memory_snapshot(device: Any) -> Any:
    """Return JSON-safe allocator statistics or a diagnostic error record."""
    try:
        return _jsonable(device.memory_stats())
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}


def _environment(jax: Any, eqx: Any, np: Any) -> dict[str, Any]:
    """Capture runtime, device, and compiler-affecting environment details."""
    devices = []
    for device in jax.devices():
        record = {
            "string": str(device),
            "id": getattr(device, "id", None),
            "platform": getattr(device, "platform", None),
            "device_kind": getattr(device, "device_kind", None),
            "process_index": getattr(device, "process_index", None),
        }
        record["memory_stats"] = _device_memory_snapshot(device)
        devices.append(record)
    fixed_environment = (
        "CUDA_VISIBLE_DEVICES",
        "LD_LIBRARY_PATH",
        "NVIDIA_TF32_OVERRIDE",
        "PATH",
        "XLA_FLAGS",
    )
    relevant_environment = {key: os.environ.get(key) for key in fixed_environment}
    relevant_environment.update(
        {
            key: value
            for key, value in sorted(os.environ.items())
            if key.startswith("JAX_") and key not in relevant_environment
        }
    )
    nvidia_smi = _command_provenance(("nvidia-smi",))
    try:
        jax_environment_info = jax.print_environment_info(return_string=True)
    except Exception as error:
        jax_environment_info = f"unavailable: {type(error).__name__}: {error}"
    try:
        x64_enabled = bool(jax.config.read("jax_enable_x64"))
    except Exception:
        x64_enabled = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "jax": getattr(jax, "__version__", importlib.metadata.version("jax")),
        "jaxlib": importlib.metadata.version("jaxlib"),
        "equinox": getattr(eqx, "__version__", importlib.metadata.version("equinox")),
        "numpy": np.__version__,
        "backend": jax.default_backend(),
        "x64_enabled": x64_enabled,
        "devices": devices,
        "environment": relevant_environment,
        "cuda_packages": _installed_package_versions(),
        "nvidia_driver_version": _nvidia_driver_version(nvidia_smi),
        "nvidia_smi": nvidia_smi,
        "jax_environment_info": jax_environment_info,
    }


def _device_memory_stats(jax: Any) -> list[dict[str, Any]]:
    """Return allocator statistics after a case when the backend exposes them."""
    records = []
    for device in jax.devices():
        records.append({"device": str(device), "stats": _device_memory_snapshot(device)})
    return records


def _resolve_dtype(jnp: Any, value: Any) -> Any:
    """Resolve JSON-friendly dtype names for configuration construction."""
    if not isinstance(value, str):
        return value
    aliases = {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
    }
    return aliases.get(value, value)


def _make_config(
    config_class: Any, jnp: Any, case: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """Construct the canonical config using only fields supported by a target checkout."""
    requested = {**CANONICAL_CONFIG, **case.get("config_overrides", {})}
    requested["use_checkpoint"] = case["suite"] == "training"
    fields = {field.name for field in dataclasses.fields(config_class)}
    applied = {
        name: _resolve_dtype(jnp, value) for name, value in requested.items() if name in fields
    }
    config = config_class(**applied)
    resolved = {}
    for field in dataclasses.fields(config):
        value = getattr(config, field.name)
        if field.name.endswith("dtype"):
            value = str(jnp.dtype(value))
        resolved[field.name] = _jsonable(value)
    manifest = {
        "profile": "canonical_cross_revision_tied",
        "requested": requested,
        "applied_fields": sorted(applied),
        "omitted_unsupported_fields": sorted(set(requested) - fields),
        "resolved": resolved,
    }
    return config, manifest


def _model_manifest(jax: Any, eqx: Any, model: Any) -> dict[str, Any]:
    """Count array leaves, elements, bytes, and dtypes in an Equinox model."""
    dtype_elements: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    array_leaves = 0
    inexact_leaves = 0
    inexact_elements = 0
    inexact_bytes = 0
    for leaf in jax.tree.leaves(model):
        if not eqx.is_array(leaf):
            continue
        array_leaves += 1
        dtype = str(leaf.dtype)
        elements = int(leaf.size)
        size_bytes = elements * int(leaf.dtype.itemsize)
        dtype_elements[dtype] = dtype_elements.get(dtype, 0) + elements
        dtype_bytes[dtype] = dtype_bytes.get(dtype, 0) + size_bytes
        if eqx.is_inexact_array(leaf):
            inexact_leaves += 1
            inexact_elements += elements
            inexact_bytes += size_bytes
    return {
        "array_leaves": array_leaves,
        "inexact_array_leaves": inexact_leaves,
        "parameter_count": inexact_elements,
        "parameter_bytes": inexact_bytes,
        "elements_by_dtype": dtype_elements,
        "bytes_by_dtype": dtype_bytes,
    }


def _comparability_manifest(
    model: Any,
    config_manifest: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    """Make output-topology limits explicit for historical comparisons."""
    requested_tied = bool(config_manifest["requested"].get("share_emb", False))
    has_output_head = hasattr(model, "tied")
    actual_tied = bool(model.tied) if has_output_head else None
    share_field_supported = "share_emb" in config_manifest["applied_fields"]
    topology_matches = actual_tied == requested_tied if has_output_head else True
    operation = str(case["operation"])
    topology_sensitive = case["suite"] == "training" or operation == "lm_ttft"
    if not has_output_head:
        note = "core model has no output head; cache-path timing is topology-independent"
    elif topology_matches:
        note = "requested and realized embedding/output topology match"
    elif not share_field_supported:
        note = (
            "target predates explicit share_emb and infers tying from output shape; "
            "attention/CEMA inference-core timings remain comparable, while LM TTFT and "
            "training parameter topology are not apples-to-apples"
        )
    else:
        note = "realized embedding/output topology differs from the benchmark request"
    return {
        "requested_share_emb": requested_tied,
        "realized_tied": actual_tied,
        "has_output_head": has_output_head,
        "explicit_share_emb_supported": share_field_supported,
        "output_topology_matches_request": topology_matches,
        "inference_core_comparable": True,
        "lm_ttft_topology_comparable": topology_matches,
        "training_topology_comparable": topology_matches,
        "this_case_topology_sensitive": topology_sensitive,
        "this_case_topology_comparable": not topology_sensitive or topology_matches,
        "eligible_for_cross_revision_ratio": not topology_sensitive or topology_matches,
        "note": note,
    }


def _completed_case_status(
    correctness_passed: bool,
    comparability: Mapping[str, Any],
) -> Status:
    """Classify a completed case without presenting incomparable timing as passed."""
    if not correctness_passed:
        return "failed"
    if not comparability["this_case_topology_comparable"]:
        return "completed_noncomparable"
    return "passed"


def _device_inputs(
    jax: Any, jnp: Any, np: Any, case: Mapping[str, Any], vocab_size: int
) -> dict[str, Any]:
    """Create deterministic host inputs and place them on the default device."""
    batch = int(case["batch_size"])
    length = int(case["sequence_length"])
    seed = int(case["seed"])
    rng = np.random.default_rng(seed)
    prefix = jax.device_put(
        jnp.asarray(rng.integers(0, vocab_size, size=(batch, length), dtype=np.int32))
    )
    continuation = jax.device_put(
        jnp.asarray(rng.integers(0, vocab_size, size=(batch, 37), dtype=np.int32))
    )
    decode = jax.device_put(
        jnp.asarray(rng.integers(0, vocab_size, size=(batch, 1), dtype=np.int32))
    )
    labels = jax.device_put(
        jnp.asarray(rng.integers(0, vocab_size, size=(batch, length), dtype=np.int32))
    )
    all_true = jax.device_put(jnp.ones((batch, length), dtype=jnp.bool_))
    # Deliberately unaligned packed runs exercise reset logic independently of
    # attention chunk boundaries. Raw IDs repeat non-adjacently; semantics are
    # defined by contiguous runs, not global uniqueness of an integer label.
    run_lengths = (509, 769, 257, 1021, 383)
    raw_ids = (1, 2, 1, 3, 2)
    segment_ids_host = np.empty((batch, length), dtype=np.int32)
    for row in range(batch):
        rotation = row % len(run_lengths)
        row_lengths = run_lengths[rotation:] + run_lengths[:rotation]
        row_ids = raw_ids[rotation:] + raw_ids[:rotation]
        start = 0
        run_index = 0
        while start < length:
            width = min(row_lengths[run_index % len(row_lengths)], length - start)
            segment_ids_host[row, start : start + width] = row_ids[run_index % len(row_ids)]
            start += width
            run_index += 1
    segment_ids = jax.device_put(jnp.asarray(segment_ids_host, dtype=jnp.int32))
    _block_tree(jax, (prefix, continuation, decode, labels, all_true, segment_ids))
    return {
        "prefix": prefix,
        "continuation": continuation,
        "decode": decode,
        "labels": labels,
        "all_true": all_true,
        "segment_ids": segment_ids,
        "segment_ids_host": segment_ids_host,
    }


def _lower_compile_measure(
    jax: Any,
    function: Any,
    arguments: tuple[Any, ...],
    warmups: int,
    iterations: int,
    profile_dir: Path | None = None,
) -> tuple[dict[str, Any], Any]:
    """Lower, compile, analyze, and measure an Equinox-filtered callable."""
    started = time.perf_counter_ns()
    lowered = function.lower(*arguments)
    lower_ms = (time.perf_counter_ns() - started) / 1e6
    started = time.perf_counter_ns()
    compiled = lowered.compile()
    compile_ms = (time.perf_counter_ns() - started) / 1e6
    compiler = _compiler_metadata(lowered, compiled)
    timing, result = _measure(
        jax,
        compiled,
        arguments,
        warmups,
        iterations,
        profile_dir,
    )
    return {
        "lower_ms": lower_ms,
        "compile_ms": compile_ms,
        "compiler": compiler,
        "timing": timing,
    }, result


def _array_error(
    jax: Any, jnp: Any, actual: Any, expected: Any, atol: float, rtol: float
) -> dict[str, Any]:
    """Report finite/allclose and exact max error statistics on device."""
    if actual.shape != expected.shape:
        return {
            "passed": False,
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }

    @jax.jit
    def summarize(left: Any, right: Any) -> tuple[Any, Any, Any, Any, Any, Any]:
        left = left.astype(jnp.float32)
        right = right.astype(jnp.float32)
        absolute = jnp.abs(left - right)
        relative = absolute / jnp.maximum(jnp.abs(right), jnp.asarray(1e-8, dtype=jnp.float32))
        finite = jnp.all(jnp.isfinite(left)) & jnp.all(jnp.isfinite(right))
        close = jnp.all(absolute <= atol + rtol * jnp.abs(right))
        rmse = jnp.sqrt(jnp.mean(jnp.square(absolute)))
        reference_rms = jnp.sqrt(jnp.mean(jnp.square(right)))
        return finite, close, jnp.max(absolute), jnp.max(relative), rmse, reference_rms

    finite, close, max_abs, max_rel, rmse, reference_rms = summarize(actual, expected)
    _block_tree(jax, (finite, close, max_abs, max_rel, rmse, reference_rms))
    return {
        "passed": bool(finite) and bool(close),
        "shape_match": True,
        "finite": bool(finite),
        "allclose": bool(close),
        "atol": atol,
        "rtol": rtol,
        "max_abs_error": float(max_abs),
        "max_relative_error": float(max_rel),
        "rmse": float(rmse),
        "reference_rms": float(reference_rms),
        "normalized_rmse": float(rmse / jnp.maximum(reference_rms, 1e-8)),
    }


def _tree_summary(jax: Any, jnp: Any, eqx: Any, tree: Any) -> dict[str, Any]:
    """Summarize finiteness and norm of all inexact array leaves."""
    leaves = [leaf for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf)]
    if not leaves:
        return {"finite": True, "inexact_leaves": 0, "elements": 0, "l2_norm": 0.0}

    def summarize(*arrays: Any) -> tuple[Any, Any]:
        finite = jnp.asarray(True)
        squared = jnp.zeros((), dtype=jnp.float32)
        for array in arrays:
            as_f32 = jnp.abs(array).astype(jnp.float32)
            finite = finite & jnp.all(jnp.isfinite(as_f32))
            squared = squared + jnp.sum(as_f32 * as_f32)
        return finite, jnp.sqrt(squared)

    finite, norm = jax.jit(summarize)(*leaves)
    _block_tree(jax, (finite, norm))
    return {
        "finite": bool(finite),
        "inexact_leaves": len(leaves),
        "elements": sum(int(leaf.size) for leaf in leaves),
        "l2_norm": float(norm),
    }


def _cache_counts(cache: Any, jax: Any, np: Any) -> list[dict[str, Any]]:
    """Extract every cache ``count`` and ``position`` field with its path."""
    records: list[dict[str, Any]] = []

    def visit(value: Any, path: str) -> None:
        if value is None:
            return
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                child = getattr(value, field.name)
                child_path = f"{path}.{field.name}" if path else field.name
                if field.name in {"count", "position"} and hasattr(child, "shape"):
                    records.append(
                        {
                            "path": child_path,
                            "dtype": str(child.dtype),
                            "shape": list(child.shape),
                            "values": _jsonable(np.asarray(jax.device_get(child))),
                        }
                    )
                else:
                    visit(child, child_path)
            return
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]")

    visit(cache, "cache")
    return records


def _count_correctness(
    records: Sequence[Mapping[str, Any]], expected: int, np: Any
) -> dict[str, Any]:
    """Check every discovered cache counter against an exact expected position."""
    mismatches = []
    for record in records:
        values = np.asarray(record["values"])
        if not np.all(values == expected):
            mismatches.append(record)
    return {
        "passed": bool(records) and not mismatches,
        "expected": expected,
        "records": list(records),
        "mismatches": mismatches,
    }


def _inference_case(
    case: Mapping[str, Any],
    jax: Any,
    jnp: Any,
    np: Any,
    eqx: Any,
    model: Any,
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark and verify one inference operation."""
    operation = case["operation"]
    prefix = inputs["prefix"]
    continuation = inputs["continuation"]
    decode = inputs["decode"]
    setup_ms = 0.0
    cache = None

    if operation in {"continuation_37", "decode_1"}:
        setup = eqx.filter_jit(lambda candidate, tokens: candidate(tokens, return_cache=True))
        started = time.perf_counter_ns()
        _, cache = setup(model, prefix)
        _block_tree(jax, cache)
        setup_ms = (time.perf_counter_ns() - started) / 1e6

    if operation == "noncached":
        function = eqx.filter_jit(lambda candidate, tokens: candidate(tokens, return_cache=False))
        arguments = (model, prefix)
        expected_count = None
    elif operation == "pristine_prefill":
        function = eqx.filter_jit(lambda candidate, tokens: candidate(tokens, return_cache=True))
        arguments = (model, prefix)
        expected_count = int(prefix.shape[1])
    elif operation in {"continuation_37", "decode_1"}:
        continuation_tokens = continuation if operation == "continuation_37" else decode
        function = eqx.filter_jit(
            lambda candidate, tokens, state: candidate(tokens, cache=state, return_cache=True)
        )
        arguments = (model, continuation_tokens, cache)
        expected_count = int(prefix.shape[1] + continuation_tokens.shape[1])
    elif operation == "lm_ttft":

        def ttft(candidate: Any, tokens: Any) -> tuple[Any, Any, Any]:
            logits, state = candidate(tokens, return_cache=True)
            last_logits = logits[:, -1, :]
            return last_logits, jnp.argmax(last_logits, axis=-1), state

        function = eqx.filter_jit(ttft)
        arguments = (model, prefix)
        expected_count = int(prefix.shape[1])
    else:  # pragma: no cover - supervisor validates the operation
        raise ValueError(f"unsupported inference operation: {operation}")

    metrics, result = _lower_compile_measure(
        jax,
        function,
        arguments,
        int(case["warmups"]),
        int(case["iterations"]),
        _case_profile_dir(case),
    )
    if operation == "lm_ttft":
        actual_logits, actual_token, result_cache = result
    else:
        actual_logits, result_cache = result
        actual_token = None

    if operation == "noncached":
        comparison = {
            "passed": bool(jnp.all(jnp.isfinite(actual_logits))),
            "reference": "self_finite_only",
        }
    else:
        if operation in {"pristine_prefill", "lm_ttft"}:
            reference_tokens = prefix
            tail = 1 if operation == "lm_ttft" else int(prefix.shape[1])
        else:
            continuation_tokens = continuation if operation == "continuation_37" else decode
            reference_tokens = jnp.concatenate((prefix, continuation_tokens), axis=1)
            tail = int(continuation_tokens.shape[1])
        reference_function = eqx.filter_jit(
            lambda candidate, tokens: candidate(tokens, return_cache=False)[0][:, -tail:, :]
        )
        expected_logits = reference_function(model, reference_tokens)
        _block_tree(jax, expected_logits)
        comparable_actual = actual_logits[:, None, :] if operation == "lm_ttft" else actual_logits
        comparison = _array_error(
            jax,
            jnp,
            comparable_actual,
            expected_logits,
            float(case["atol"]),
            float(case["rtol"]),
        )
        comparison["reference"] = "vectorized_noncached_full_sequence"
        if actual_token is not None:
            expected_token = jnp.argmax(expected_logits[:, -1, :], axis=-1)
            token_equal = bool(jnp.all(actual_token == expected_token))
            comparison["argmax_token_equal"] = token_equal
            comparison["passed"] = comparison["passed"] and token_equal

    cache_summary = _tree_summary(jax, jnp, eqx, result_cache)
    if expected_count is None:
        counts = {"passed": result_cache is None, "expected": None, "records": []}
    else:
        count_records = _cache_counts(result_cache, jax, np)
        counts = _count_correctness(count_records, expected_count, np)
    correctness = {
        "passed": bool(comparison["passed"] and cache_summary["finite"] and counts["passed"]),
        "logits": comparison,
        "cache": cache_summary,
        "counts": counts,
    }
    metrics["setup_prefill_ms"] = setup_ms
    return metrics, correctness


def _training_case(
    case: Mapping[str, Any],
    jax: Any,
    jnp: Any,
    eqx: Any,
    model: Any,
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark checkpointed loss forward or forward-plus-backward."""
    mode = case["mode"]
    mask = inputs["all_true"] if mode == "all_true" else None
    segments = inputs["segment_ids"] if mode == "packed" else None
    tokens = inputs["prefix"]
    labels = inputs["labels"]
    loss_chunk_size = case.get("loss_chunk_size")

    def loss_function(
        candidate: Any,
        token_ids: Any,
        target_ids: Any,
        attention_mask: Any,
        segment_ids: Any,
        dropout_key: Any,
        deterministic: bool,
    ) -> Any:
        loss_kwargs = {} if loss_chunk_size is None else {"loss_chunk_size": int(loss_chunk_size)}
        return candidate.compute_loss(
            token_ids,
            target_ids,
            attention_mask=attention_mask,
            segment_ids=segment_ids,
            deterministic=deterministic,
            key=dropout_key,
            **loss_kwargs,
        )

    if case["operation"] == "forward":
        function = eqx.filter_jit(loss_function)
    elif case["operation"] == "forward_backward":
        function = eqx.filter_jit(eqx.filter_value_and_grad(loss_function))
    else:  # pragma: no cover - supervisor validates the operation
        raise ValueError(f"unsupported training operation: {case['operation']}")
    dropout_key = jax.random.fold_in(jax.random.PRNGKey(int(case["seed"])), 1)
    arguments = (model, tokens, labels, mask, segments, dropout_key, False)
    metrics, result = _lower_compile_measure(
        jax,
        function,
        arguments,
        int(case["warmups"]),
        int(case["iterations"]),
        _case_profile_dir(case),
    )
    if case["operation"] == "forward_backward":
        loss, gradients = result
        gradient_summary = _tree_summary(jax, jnp, eqx, gradients)
    else:
        loss = result
        gradient_summary = None
    loss_value = float(loss)
    loss_finite = math.isfinite(loss_value)

    reference: dict[str, Any] = {"kind": None, "max_abs_error": None, "passed": True}
    if mode == "all_true":
        plain = eqx.filter_jit(loss_function)(model, tokens, labels, None, None, dropout_key, False)
        _block_tree(jax, plain)
        error = abs(loss_value - float(plain))
        tolerance = float(case["atol"]) + float(case["rtol"]) * abs(float(plain))
        reference = {
            "kind": "same_batch_attention_mask_none",
            "expected_loss": float(plain),
            "max_abs_error": error,
            "tolerance": tolerance,
            "passed": error <= tolerance,
        }
    elif mode == "packed":
        config = getattr(model, "config", None)
        reference_deterministic = any(
            float(getattr(config, name, 0.0)) > 0.0
            for name in ("dropout", "attention_dropout", "hidden_dropout")
        )
        if reference_deterministic:
            packed_reference_loss = eqx.filter_jit(loss_function)(
                model,
                tokens,
                labels,
                mask,
                segments,
                dropout_key,
                True,
            )
            _block_tree(jax, packed_reference_loss)
            reference_loss_value = float(packed_reference_loss)
        else:
            reference_loss_value = loss_value

        # Recover actual contiguous runs from host segment IDs. Raw identifiers
        # deliberately recur in non-adjacent runs, and each batch row rotates
        # the unaligned run-width pattern. Group equal widths so the independent
        # reference needs one compilation per distinct shape rather than one
        # per document.
        segment_ids_host = inputs["segment_ids_host"]
        grouped_runs: dict[int, list[tuple[int, int, int]]] = {}
        row_run_lengths: list[list[int]] = []
        for row in range(int(tokens.shape[0])):
            boundaries = [0]
            for index in range(1, int(tokens.shape[1])):
                if segment_ids_host[row, index] != segment_ids_host[row, index - 1]:
                    boundaries.append(index)
            boundaries.append(int(tokens.shape[1]))
            lengths = []
            for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
                width = stop - start
                lengths.append(width)
                grouped_runs.setdefault(width, []).append((row, start, stop))
            row_run_lengths.append(lengths)

        weighted_loss = 0.0
        transitions = 0
        independent = eqx.filter_jit(loss_function)
        for width, runs in sorted(grouped_runs.items()):
            if width <= 1:
                continue
            document_tokens = jnp.stack(
                [tokens[row, start:stop] for row, start, stop in runs],
                axis=0,
            )
            document_labels = jnp.stack(
                [labels[row, start:stop] for row, start, stop in runs],
                axis=0,
            )
            document_loss = independent(
                model,
                document_tokens,
                document_labels,
                None,
                None,
                dropout_key,
                reference_deterministic,
            )
            _block_tree(jax, document_loss)
            weight = len(runs) * (width - 1)
            weighted_loss += float(document_loss) * weight
            transitions += weight
        expected = weighted_loss / max(transitions, 1)
        error = abs(reference_loss_value - expected)
        tolerance = float(case["atol"]) + float(case["rtol"]) * abs(expected)
        reference = {
            "kind": "transition_weighted_independent_segments",
            "observed_loss": reference_loss_value,
            "expected_loss": expected,
            "max_abs_error": error,
            "tolerance": tolerance,
            "passed": error <= tolerance,
            "row_run_lengths": row_run_lengths,
            "deterministic_auxiliary": reference_deterministic,
        }

    gradients_pass = gradient_summary is None or gradient_summary["finite"]
    correctness = {
        "passed": bool(loss_finite and gradients_pass and reference["passed"]),
        "loss": loss_value,
        "loss_finite": loss_finite,
        "gradients": gradient_summary,
        "reference": reference,
        "checkpointing_requested": True,
        "deterministic": False,
    }
    return metrics, correctness


def _worker(spec_path: Path, result_path: Path) -> int:
    """Execute one benchmark case after binding imports to its source root."""
    case = json.loads(spec_path.read_text(encoding="utf-8"))
    repo = Path(case["repo_root"]).resolve()
    sys.path.insert(0, str(repo / "src"))
    started_at = _utc_now()
    worker_started = time.perf_counter()
    base: dict[str, Any] = {
        **case,
        "status": "failed",
        "started_at": started_at,
        "finished_at": None,
        "provenance": _repo_provenance(repo),
        "benchmark_script": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp
        import numpy as np

        from megalodon_jax.config import MegalodonConfig
        from megalodon_jax.model import MegalodonForCausalLM, MegalodonModel

        base["environment"] = _environment(jax, eqx, np)
        config, config_manifest = _make_config(MegalodonConfig, jnp, case)
        base["config"] = config_manifest
        model_started = time.perf_counter_ns()
        use_lm_wrapper = case["suite"] == "training" or case["operation"] == "lm_ttft"
        model_class = MegalodonForCausalLM if use_lm_wrapper else MegalodonModel
        model = model_class(config, key=jax.random.PRNGKey(int(case["seed"])))
        model = jax.device_put(model)
        _block_tree(jax, model)
        base["model_initialization_ms"] = (time.perf_counter_ns() - model_started) / 1e6
        base["model"] = _model_manifest(jax, eqx, model)
        base["model"]["class"] = model_class.__name__
        base["comparability"] = _comparability_manifest(model, config_manifest, case)
        inputs = _device_inputs(jax, jnp, np, case, int(config.vocab_size))
        if case["suite"] == "inference":
            metrics, correctness = _inference_case(case, jax, jnp, np, eqx, model, inputs)
        elif case["suite"] == "training":
            metrics, correctness = _training_case(case, jax, jnp, eqx, model, inputs)
        else:
            raise ValueError(f"unsupported suite: {case['suite']}")
        metrics["runtime_device_memory"] = _device_memory_stats(jax)
        base["metrics"] = metrics
        base["correctness"] = correctness
        base["status"] = _completed_case_status(correctness["passed"], base["comparability"])
        if not correctness["passed"]:
            base["error"] = {
                "type": "CorrectnessFailure",
                "message": "timed execution completed but correctness checks failed",
            }
    except Exception as error:
        rendered = "".join(traceback.format_exception(type(error), error, error.__traceback__))
        base["status"] = _classify_failure(rendered)
        base["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": rendered,
        }
    finally:
        base["finished_at"] = _utc_now()
        base["worker_wall_seconds"] = time.perf_counter() - worker_started
        _atomic_json(result_path, base)
    print(f"{case['case_id']}: {base['status']}")
    return 0 if base["status"] in {"passed", "completed_noncomparable"} else 1


def _parser() -> argparse.ArgumentParser:
    """Build the supervisor/worker command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        action="append",
        help="Target NAME=PATH; repeat for current and historical worktrees (default: current=.)",
    )
    parser.add_argument("--suite", action="append", choices=("inference", "training"))
    parser.add_argument(
        "--inference-operations",
        type=_csv_strings,
        default=INFERENCE_OPERATIONS,
        help=f"Comma-separated operations (default: {','.join(INFERENCE_OPERATIONS)})",
    )
    parser.add_argument(
        "--training-operations",
        type=_csv_strings,
        default=TRAINING_OPERATIONS,
        help=f"Comma-separated operations (default: {','.join(TRAINING_OPERATIONS)})",
    )
    parser.add_argument(
        "--training-modes",
        type=_csv_strings,
        default=TRAINING_MODES,
        help=f"Comma-separated modes (default: {','.join(TRAINING_MODES)})",
    )
    parser.add_argument(
        "--inference-lengths",
        type=_csv_positive_ints,
        default=(64, 512, 2048, 4096),
    )
    parser.add_argument("--inference-batches", type=_csv_positive_ints, default=(1,))
    parser.add_argument("--training-lengths", type=_csv_positive_ints, default=(2048, 4096))
    parser.add_argument("--training-batches", type=_csv_positive_ints, default=(1, 2, 4))
    parser.add_argument(
        "--loss-chunk-size",
        type=int,
        help="Opt-in token chunk size for the memory-bounded training loss head",
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--seed", type=int, default=20250712)
    parser.add_argument("--atol", type=float, help="Override the dtype-aware absolute tolerance")
    parser.add_argument("--rtol", type=float, help="Override the dtype-aware relative tolerance")
    parser.add_argument(
        "--bf16-atol",
        type=float,
        default=8e-2,
        help="BF16 absolute envelope (about two ULP at unit-scale across deep cached graphs)",
    )
    parser.add_argument("--bf16-rtol", type=float, default=2e-2)
    parser.add_argument("--fp32-atol", type=float, default=2e-6)
    parser.add_argument("--fp32-rtol", type=float, default=2e-5)
    parser.add_argument(
        "--config-json",
        help="Inline JSON object or @path overriding canonical model fields",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local-scratch/model-path-benchmark.json"),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help=(
            "Optional XProf output root; each worker records one extra synchronized "
            "iteration outside timing in its own case directory"
        ),
    )
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-unknown-revision", action="store_true")
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--_worker-spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-result", type=Path, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    """Run the supervisor or one private worker invocation."""
    args = _parser().parse_args()
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive")
    if args.loss_chunk_size is not None and args.loss_chunk_size <= 0:
        raise ValueError("--loss-chunk-size must be positive")
    for name in ("atol", "rtol", "bf16_atol", "bf16_rtol", "fp32_atol", "fp32_rtol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args._worker_spec is not None or args._worker_result is not None:
        if args._worker_spec is None or args._worker_result is None:
            raise ValueError("private worker invocation requires both worker paths")
        return _worker(args._worker_spec.resolve(), args._worker_result.resolve())
    return _supervisor(args)


if __name__ == "__main__":
    raise SystemExit(main())
