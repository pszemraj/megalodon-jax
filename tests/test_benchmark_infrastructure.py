"""Regression tests for benchmark comparability and provenance metadata."""

from __future__ import annotations

import importlib.metadata
import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import benchmark_model_paths as benchmark

pytestmark = pytest.mark.fast


def test_cross_revision_default_uses_tied_topology() -> None:
    """The default must match historical output topology for LM timing ratios."""
    assert benchmark.CANONICAL_CONFIG["share_emb"] is True
    assert "softmax_dtype" not in benchmark.CANONICAL_CONFIG
    assert "gemm_backend" not in benchmark.CANONICAL_CONFIG

    config = {
        "requested": {"share_emb": True},
        "applied_fields": ["share_emb"],
    }
    case = {"suite": "training", "operation": "forward"}
    comparable = benchmark._comparability_manifest(
        SimpleNamespace(tied=True),
        config,
        case,
    )
    assert comparable["this_case_topology_comparable"] is True
    assert comparable["eligible_for_cross_revision_ratio"] is True
    assert benchmark._completed_case_status(True, comparable) == "passed"


def test_noncomparable_topology_has_distinct_completion_status() -> None:
    """Successful execution must not label topology-sensitive mismatches passed."""
    config = {
        "requested": {"share_emb": False},
        "applied_fields": [],
    }
    case = {"suite": "inference", "operation": "lm_ttft"}
    comparability = benchmark._comparability_manifest(
        SimpleNamespace(tied=True),
        config,
        case,
    )
    assert comparability["this_case_topology_comparable"] is False
    assert comparability["eligible_for_cross_revision_ratio"] is False
    assert benchmark._completed_case_status(True, comparability) == "completed_noncomparable"
    assert benchmark._completed_case_status(False, comparability) == "failed"


def test_cuda_package_provenance_accepts_distribution_naming_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA provenance resolves both suffixed and current unsuffixed wheel names."""
    versions = {"nvidia-cublas": "13.2.0.9", "jax-cuda13-plugin": "0.10.2"}

    def version(distribution: str) -> str:
        if distribution not in versions:
            raise importlib.metadata.PackageNotFoundError(distribution)
        return versions[distribution]

    monkeypatch.setattr(benchmark.importlib.metadata, "version", version)
    records = benchmark._installed_package_versions(
        {
            "cublas": ("nvidia-cublas-cu13", "nvidia-cublas"),
            "plugin": ("jax-cuda13-plugin",),
            "missing": ("not-installed",),
        }
    )
    assert records["cublas"]["installed"] == [
        {"distribution": "nvidia-cublas", "version": "13.2.0.9"}
    ]
    assert records["plugin"]["installed"][0]["version"] == "0.10.2"
    assert records["missing"]["installed"] == []


def test_driver_version_is_extracted_without_requiring_nvidia_smi() -> None:
    """Driver parsing is useful when present and harmless on CPU-only hosts."""
    record = {
        "available": True,
        "stdout": "NVIDIA-SMI 595.71.05    Driver Version: 595.71.05    CUDA Version: 13.2",
    }
    assert benchmark._nvidia_driver_version(record) == "595.71.05"
    assert benchmark._nvidia_driver_version({"available": False}) is None


def test_source_archives_have_a_nonzero_version_fallback() -> None:
    """Raw source archives must not silently build packages as version 0.0.0."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    configuration = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    fallback = configuration["tool"]["setuptools_scm"]["fallback_version"]
    assert fallback != "0.0.0"


def test_supervisor_discards_stale_worker_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashed rerun cannot inherit a prior worker's passing evidence."""
    output = tmp_path / "benchmark.json"
    case = {
        "case_id": "current--training--forward--plain--b1--l8",
        "repo_root": str(tmp_path),
    }
    result_path = tmp_path / "benchmark.cases" / f"{case['case_id']}.json"
    benchmark._atomic_json(result_path, {**case, "status": "passed"})
    monkeypatch.setattr(benchmark, "_build_cases", lambda args, repos: [case])
    monkeypatch.setattr(
        benchmark,
        "_repo_provenance",
        lambda path: {"revision": "test", "dirty": False, "dirty_entries": []},
    )
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="crashed"),
    )
    args = SimpleNamespace(
        repo=None,
        allow_unknown_revision=False,
        allow_dirty=False,
        output=output,
        dry_run=False,
        timeout_seconds=10.0,
        warmups=0,
        iterations=1,
        profile_dir=None,
        allow_failures=False,
    )

    exit_code = benchmark._supervisor(args)

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["summary"]["passed"] == 0
    assert report["summary"]["failed"] == 1
    assert report["cases"][0]["status"] == "failed"
    assert report["cases"][0]["error"]["type"] == "WorkerProcessError"


def test_training_case_reuses_stable_dropout_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Timed and reference stochastic losses receive the same explicit key."""
    observed_keys: list[np.ndarray] = []

    class DropoutModel:
        def compute_loss(
            self,
            *args: object,
            deterministic: bool,
            key: object,
            **kwargs: object,
        ) -> jax.Array:
            del args, kwargs
            assert deterministic is False
            assert key is not None
            observed_keys.append(np.asarray(key))
            return jnp.asarray(1.25, dtype=jnp.float32)

    def measure_once(
        jax_module: object,
        function: object,
        arguments: tuple[object, ...],
        warmups: int,
        iterations: int,
        profile_dir: Path | None,
    ) -> tuple[dict[str, object], object]:
        del jax_module, warmups, iterations, profile_dir
        return {}, function(*arguments)

    monkeypatch.setattr(benchmark, "_lower_compile_measure", measure_once)
    case = {
        "mode": "all_true",
        "operation": "forward",
        "seed": 1729,
        "warmups": 0,
        "iterations": 1,
        "atol": 0.0,
        "rtol": 0.0,
    }
    tokens = jnp.asarray([[1, 2, 3]], dtype=jnp.int32)
    inputs = {
        "prefix": tokens,
        "labels": tokens,
        "all_true": jnp.ones_like(tokens, dtype=jnp.bool_),
    }
    fake_eqx = SimpleNamespace(filter_jit=lambda function: function)

    _, correctness = benchmark._training_case(
        case,
        jax,
        jnp,
        fake_eqx,
        DropoutModel(),
        inputs,
    )

    assert correctness["passed"] is True
    assert len(observed_keys) == 2
    np.testing.assert_array_equal(observed_keys[0], observed_keys[1])
