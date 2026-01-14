"""Helpers for importing the external megalodon-hf package in tests."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_external(module: ModuleType, name: str) -> ModuleType:
    module_path = getattr(module, "__file__", None)
    if module_path is None:
        return module
    resolved = Path(module_path).resolve()
    if _PROJECT_ROOT in resolved.parents:
        raise RuntimeError(f"{name} resolved inside this repo ({resolved}).")
    return module


def _import_external(name: str) -> ModuleType:
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("megalodon"):
            raise RuntimeError(
                "megalodon-hf is not installed. Install the dev dependency: "
                "`pip install -e '.[dev]'`."
            ) from exc
        raise
    return _ensure_external(module, name)


def megalodon() -> object:
    """Import the external megalodon package."""
    return _import_external("megalodon")


def modeling() -> object:
    """Import megalodon.modeling_megalodon from the external package."""
    _import_external("megalodon")
    return _import_external("megalodon.modeling_megalodon")
