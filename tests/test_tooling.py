"""Tests for standalone verification and benchmark tooling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import verify_modeling_correctness, verify_timestep_norm_candidates


@pytest.mark.parametrize(
    "module",
    [verify_modeling_correctness, verify_timestep_norm_candidates],
)
def test_git_revision_accepts_only_requested_checkout_root(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source archive nested in another checkout cannot inherit its revision."""
    source = tmp_path / "source-archive"
    source.mkdir()
    parent_revision = "a" * 40

    def enclosing_repo(*args, **kwargs):
        return SimpleNamespace(stdout=f"{tmp_path}\n{parent_revision}\n")

    monkeypatch.setattr(module.subprocess, "run", enclosing_repo)
    monkeypatch.setenv("MEGALODON_JAX_SOURCE_REVISION", "archive-revision")
    assert module._git_revision(source) == "archive-revision"

    def exact_repo(*args, **kwargs):
        return SimpleNamespace(stdout=f"{source}\n{parent_revision}\n")

    monkeypatch.setattr(module.subprocess, "run", exact_repo)
    assert module._git_revision(source) == parent_revision
